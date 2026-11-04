//! Window ends: `_prepare_slot_for_alignment`, `_batch_align_slots` (with the
//! batched DTW hand-off) and `_finalize_slot_after_alignment`

use eyre::{Result, ensure};

use super::Transcriber;
use super::model::{Model, N_FRAMES};
use super::probe::Event;
use super::trace;
use crate::decode::Status;
use crate::dtw::{DtwOptions, TokenizerDtw, fill_word_timings_batch};
use crate::post::timings::TIMINGS_FORMAT;
use crate::post::{
    FinalizeState, append_window, clean_chunk, cut_after_last_timed_word, finalize_transcription,
    offset_timestamps, py, strip_to_last_closing,
};

/// Mel frames per second (`FRAMES_PER_SECOND`)
const FRAMES_PER_SECOND: f64 = 100.0;

/// Windows a chunk may continue into before it is forced to finish
const MAX_CONTINUATIONS: u32 = 20;

/// Least a continuing window advances, in seconds
const MIN_ADVANCE: f64 = 1.0;

/// Which kind of window end a slot is at
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// `status == "process"`: the window stopped early and may continue
    Process,
    /// `status == "finished"`: the chunk is done
    Finished,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Process => "process",
            Self::Finished => "finished",
        }
    }
}

/// One slot queued for alignment (the dict `_prepare_slot_for_alignment` returns)
#[derive(Debug, Clone, PartialEq)]
pub struct AlignItem {
    pub slot: usize,
    /// The finished result of a skipped (non-English) slot, which needs no alignment
    pub skip: Option<ChunkResult>,
    /// The window's decoder text
    pub cur: String,
    /// Cut the text after its last timed word before aligning (when continuing)
    pub clip_to_last_timing: bool,
    /// The chunk continues into another window
    pub more: bool,
    pub phase: Phase,
    /// Where the window ends when a repetition stop skipped ahead
    pub last_timing_override: Option<f64>,
}

/// A finished chunk (the result dict)
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkResult {
    pub id: String,
    /// The cleaned text with the chunk offset applied (`prediction`, also `transcription`)
    pub prediction: String,
    /// `raw_cur`: the slot's window texts before cleaning; `None` for skipped chunks
    pub raw_cur: Option<String>,
    pub language: String,
    pub stop_reason: String,
    /// Original speaker id of each renumbered id (`speaker_id_map`); `None` for skipped chunks
    pub speaker_id_map: Option<Vec<i64>>,
    /// `meta["offset"]`
    pub offset: Option<f64>,
    /// `meta["duration"]`
    pub duration: Option<f64>,
}

/// Python's `a or b or 0` over optional floats, where `0.0` is falsy
fn py_or(a: Option<f64>, b: Option<f64>) -> f64 {
    a.filter(|v| *v != 0.0)
        .or(b.filter(|v| *v != 0.0))
        .unwrap_or(0.0)
}

impl<M: Model> Transcriber<M> {
    /// `_prepare_slot_for_alignment(s)`: decides whether a stopped window
    /// continues (`more`) and where it ends
    pub(crate) fn prepare_slot_for_alignment(
        &mut self,
        slot_idx: usize,
    ) -> Result<Option<AlignItem>> {
        let item = self.prepare_item(slot_idx);
        emit!(
            self,
            Event::PrepareAlignment {
                slot: &self.slots[slot_idx],
                item: item.as_ref(),
            }
        );
        Ok(item)
    }

    fn prepare_item(&mut self, slot_idx: usize) -> Option<AlignItem> {
        let detect = self.config.lang.is_none();
        let s = &mut self.slots[slot_idx];
        let st = &s.state;
        if self.config.skip_non_english
            && detect
            && !st.language.is_empty()
            && st.token_count >= 2
            && st.language != "en"
        {
            let result = ChunkResult {
                id: s.id.clone().unwrap_or_default(),
                prediction: String::new(),
                raw_cur: None,
                language: st.language.clone(),
                stop_reason: format!("non_english:{}", st.language),
                speaker_id_map: None,
                offset: s.meta.offset,
                duration: s.meta.duration,
            };
            s.state.status = Status::Ready;
            return Some(AlignItem {
                slot: slot_idx,
                skip: Some(result),
                cur: String::new(),
                clip_to_last_timing: false,
                more: false,
                phase: Phase::Finished,
                last_timing_override: None,
            });
        }

        let duration = st.duration;
        let has_duration = duration != 0.0;
        match st.status {
            Status::Process => {
                let mut last_timing = py_or(st.last_timing, None);
                let stop_reason = st.stop_reason.as_deref().unwrap_or("");
                let repetition = stop_reason.contains("repetition");
                let more = if (stop_reason.contains("eot_predicted") && st.last_timing.is_none())
                    || (repetition
                        && (stop_reason.contains("[pause") || stop_reason.contains("[breath")))
                {
                    // EOT with no timing, or a pause/breath loop: nothing more to say
                    false
                } else if repetition {
                    // a speech loop: skip ahead a second and try to continue
                    let cap = if has_duration {
                        duration - st.offset
                    } else {
                        last_timing + 1.0
                    };
                    last_timing = py::min(last_timing + 1.0, cap);
                    duration - (st.offset + last_timing) > 0.5
                } else {
                    // continue while more than half a second is left after the last timing
                    duration - (st.offset + last_timing) > 0.5
                };

                Some(AlignItem {
                    slot: slot_idx,
                    skip: None,
                    cur: st.cur.clone(),
                    clip_to_last_timing: more,
                    more,
                    phase: Phase::Process,
                    last_timing_override: repetition.then_some(last_timing),
                })
            }
            Status::Finished => Some(AlignItem {
                slot: slot_idx,
                skip: None,
                cur: st.cur.clone(),
                clip_to_last_timing: false,
                more: false,
                phase: Phase::Finished,
                last_timing_override: None,
            }),
            _ => None,
        }
    }

    /// `_batch_align_slots(items)`: per-slot window ends, one batched DTW over every
    /// aligning slot, then finalisation. Returns one result per item (`None` while
    /// a chunk continues)
    pub(crate) fn batch_align_slots(
        &mut self,
        items: Vec<AlignItem>,
    ) -> Result<Vec<Option<ChunkResult>>> {
        let slots: Vec<usize> = items.iter().map(|i| i.slot).collect();
        let mut results: Vec<Option<ChunkResult>> = vec![None; items.len()];
        let mut to_align = Vec::new();
        for (i, item) in items.into_iter().enumerate() {
            match item.skip {
                Some(r) => results[i] = Some(r),
                None => to_align.push((i, item)),
            }
        }

        if !to_align.is_empty() {
            self.align_and_finalize(&to_align, &mut results)?;
        }

        emit!(
            self,
            Event::BatchAlign {
                slots: &slots,
                results: &results,
            }
        );
        Ok(results)
    }

    fn align_and_finalize(
        &mut self,
        to_align: &[(usize, AlignItem)],
        results: &mut [Option<ChunkResult>],
    ) -> Result<()> {
        // each slot's text up to its last timed word when continuing, and where it ends
        let mut outs = Vec::with_capacity(to_align.len());
        let mut last_timings = Vec::with_capacity(to_align.len());
        for (_, item) in to_align {
            let s = &self.slots[item.slot];
            let fallback = || py_or(item.last_timing_override, s.state.last_timing);
            if item.clip_to_last_timing {
                let out = cut_after_last_timed_word(&item.cur)?;
                let matches = TIMINGS_FORMAT.captures_all(&out)?;
                let last = match matches.last() {
                    Some(m) => py::float(m.group(1))?,
                    None => fallback(),
                };
                outs.push(out);
                last_timings.push(last);
            } else {
                outs.push(item.cur.clone());
                last_timings.push(fallback());
            }
        }

        let _range = trace::Range::new("align");
        let pre_augs = if self.config.dtw_align {
            Some(self.batched_dtw(to_align, &outs)?)
        } else {
            None
        };

        for (k, (orig, item)) in to_align.iter().enumerate() {
            results[*orig] = self.finalize_slot_after_alignment(
                item.slot,
                &outs[k],
                last_timings[k],
                item.more,
                item.phase,
                pre_augs.as_ref().map(|p| p[k].as_str()),
            )?;
        }

        Ok(())
    }

    /// The `fill_word_timings_batch` call: window-relative texts in, texts with
    /// DTW word times out, shifted by each slot's window offset
    fn batched_dtw(
        &mut self,
        to_align: &[(usize, AlignItem)],
        outs: &[String],
    ) -> Result<Vec<String>> {
        let max_audio_length = self.config.max_audio_length;
        let mut slot_data: Vec<(String, i64)> = Vec::with_capacity(to_align.len());
        let mut xa_slots = Vec::with_capacity(to_align.len());
        for ((_, item), out) in to_align.iter().zip(outs) {
            let s = &self.slots[item.slot];
            let raw_cur = finalize_transcription(&finalize_state(s), out, false)?;
            let win_dur = match s.duration() {
                Some(d) => py::min(max_audio_length, py::max(0.0, d - s.last_offset)),
                None => max_audio_length,
            };
            let num_frames = ((win_dur * FRAMES_PER_SECOND) as i64).min(N_FRAMES as i64);
            slot_data.push((raw_cur, num_frames));
            xa_slots.push(item.slot);
        }

        let tables = self.tables.clone();
        let tokenizer = TokenizerDtw::new(&tables.tok)?;
        let model = &mut self.model;
        // dtw_prep nests inside, so idle time under this range is the host DTW
        let dtw = trace::Range::new("dtw_cpu");
        let filled =
            fill_word_timings_batch(&slot_data, &tokenizer, &DtwOptions::default(), |batch| {
                let rows: Vec<usize> = batch.slots.iter().map(|&si| xa_slots[si]).collect();
                model.dtw_prep(batch, &rows)
            })?;
        drop(dtw);
        ensure!(
            filled.len() == slot_data.len(),
            "DTW returned {} texts for {} slots",
            filled.len(),
            slot_data.len()
        );

        emit!(
            self,
            Event::FillWordTimings {
                xa_slots: &xa_slots,
                texts: &slot_data,
                out: &filled,
            }
        );

        to_align
            .iter()
            .zip(filled)
            .map(|((_, item), text)| {
                let offset = self.slots[item.slot].state.offset;
                if offset > 0.0 {
                    offset_timestamps(&text, offset)
                } else {
                    Ok(text)
                }
            })
            .collect()
    }

    /// The window's text without a batched DTW result: finalised window-relative,
    /// then shifted by the window offset (DTW off, so `_dtw_fill_cur` is the identity)
    fn window_text(&self, slot_idx: usize, out: &str) -> Result<String> {
        let s = &self.slots[slot_idx];
        let cur = finalize_transcription(&finalize_state(s), out, false)?;
        if s.state.offset > 0.0 {
            offset_timestamps(&cur, s.state.offset)
        } else {
            Ok(cur)
        }
    }

    /// `_finalize_slot_after_alignment(...)`: appends the window to the chunk's
    /// text; a `process` window either continues (returns `None`, slot back to
    /// `started` at the advanced offset) or finishes the chunk, and a finished
    /// chunk is cleaned and returned
    pub(crate) fn finalize_slot_after_alignment(
        &mut self,
        slot_idx: usize,
        out: &str,
        last_timing: f64,
        more: bool,
        phase: Phase,
        pre_augmented: Option<&str>,
    ) -> Result<Option<ChunkResult>> {
        let result = self.finalize_inner(slot_idx, out, last_timing, more, phase, pre_augmented)?;
        emit!(
            self,
            Event::Finalize {
                slot: &self.slots[slot_idx],
                out,
                last_timing,
                more,
                phase: phase.as_str(),
                pre_augmented,
                result: result.as_ref(),
            }
        );
        Ok(result)
    }

    fn finalize_inner(
        &mut self,
        slot_idx: usize,
        out: &str,
        last_timing: f64,
        more: bool,
        phase: Phase,
        pre_augmented: Option<&str>,
    ) -> Result<Option<ChunkResult>> {
        let lang_set = self.config.lang.is_some();
        let speaker_tags = self.config.decode.speaker_tags;
        let init_len = self.init_tokens.len();
        let max_audio_length = self.config.max_audio_length;

        if phase == Phase::Process {
            let cur = match pre_augmented {
                Some(c) => c.to_owned(),
                None => self.window_text(slot_idx, out)?,
            };

            let advance = if more {
                py::max(last_timing, MIN_ADVANCE)
            } else {
                last_timing
            };
            let last_token = *self.init_tokens.last().expect("non-empty start sequence");
            let s = &mut self.slots[slot_idx];
            s.state.offset += advance;
            // the next window re-covers anything after the last closing timestamp
            let cur = if more {
                strip_to_last_closing(&cur)?
            } else {
                cur
            };
            append_window(&mut s.sofar, &cur, true);
            s.state.token_count = init_len;
            s.state.last_token_id = last_token;
            s.state.cur = if lang_set && speaker_tags {
                "[0] ".to_owned()
            } else {
                String::new()
            };
            // a new window needs its fresh `[0]`, which cutearly would strip
            s.cutearly = false;
            s.state.last_timing = None;
            s.state.window_max_timing = 0.0;
            s.continuation_count += 1;

            if more && s.continuation_count < MAX_CONTINUATIONS {
                s.state.status = Status::Started;
                let row = &mut self.all_tokens[slot_idx];
                row.fill(0);
                row[..init_len].copy_from_slice(&self.init_tokens);
                self.seq_lens[slot_idx] = init_len;
                let remaining = if s.state.duration != 0.0 {
                    s.state.duration - s.state.offset
                } else {
                    0.0
                };
                s.is_final_chunk = remaining <= max_audio_length;
                return Ok(None);
            }

            if more {
                s.state.stop_reason = Some(format!("max_continuations:{}", s.continuation_count));
            }

            s.state.status = Status::Finished;
            if s.state.stop_reason.as_deref().is_none_or(str::is_empty) {
                s.state.stop_reason = Some("audio_exhausted".to_owned());
            }
        }

        if self.slots[slot_idx].status() != Status::Finished {
            return Ok(None);
        }

        if phase == Phase::Finished {
            let cur = match pre_augmented {
                Some(c) => c.to_owned(),
                None => self.window_text(slot_idx, out)?,
            };
            append_window(&mut self.slots[slot_idx].sofar, &cur, false);
        }

        let s = &mut self.slots[slot_idx];
        let cleaned = clean_chunk(&s.sofar, self.config.align_override_score)?;
        let chunk_offset = s.meta.offset.unwrap_or(0.0);
        let prediction = if chunk_offset > 0.0 {
            offset_timestamps(&cleaned.text, chunk_offset)?
        } else {
            cleaned.text
        };

        if s.state.stop_reason.as_deref().is_none_or(str::is_empty) {
            s.state.stop_reason = Some("completed".to_owned());
        }

        s.state.status = Status::Ready;
        Ok(Some(ChunkResult {
            id: s.id.clone().unwrap_or_default(),
            prediction,
            raw_cur: Some(s.sofar.clone()),
            language: s.state.language.clone(),
            stop_reason: s.state.stop_reason.clone().unwrap_or_default(),
            speaker_id_map: Some(cleaned.speaker_order),
            offset: s.meta.offset,
            duration: s.meta.duration,
        }))
    }
}

/// The fields `finalize_transcription` reads; `keepspeaker` is a class attribute
/// that is always `False`
fn finalize_state(s: &super::Slot) -> FinalizeState {
    FinalizeState {
        offset: s.state.offset,
        keepspeaker: false,
        cutearly: s.cutearly,
    }
}
