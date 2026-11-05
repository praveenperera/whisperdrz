//! Window ends: `_prepare_slot_for_alignment`, `_batch_align_slots` (with the
//! batched DTW hand-off) and `_finalize_slot_after_alignment`

use std::thread::JoinHandle;

use eyre::{Result, WrapErr, ensure, eyre};

use super::Transcriber;
use super::model::{DtwMatrices, Model, N_FRAMES};
use super::probe::Event;
use super::trace;
use crate::decode::Status;
use crate::dtw::{DtwOptions, PreparedBatch, TokenizerDtw};
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
    ///
    /// When the model queues the DTW ([`Model::dtw_launch`] not ready), every slot's
    /// state moves on now, as it doesn't depend on the aligned text, and the texts are
    /// finished later ([`Self::settle_align`]): the call then returns no results, and the
    /// whole batch's results, skipped slots included, come out in order once it lands
    pub(crate) fn batch_align_slots(
        &mut self,
        items: Vec<AlignItem>,
    ) -> Result<Vec<Option<ChunkResult>>> {
        // the caller settles the previous batch first ([`Self::settle_align`]): resolving
        // it here would put its results after this batch's if this one finishes inline
        ensure!(
            self.align_pending.is_none() && self.resolved.is_empty(),
            "alignment batch started before the previous one settled"
        );
        let slots: Vec<usize> = items.iter().map(|i| i.slot).collect();
        let mut results: Vec<Option<ChunkResult>> = vec![None; items.len()];
        let mut to_align = Vec::new();
        for (i, item) in items.into_iter().enumerate() {
            match item.skip {
                Some(r) => results[i] = Some(r),
                None => to_align.push((i, item)),
            }
        }

        if !to_align.is_empty()
            && let Some(mut queued) = self.align_and_finalize(&to_align, &mut results)?
        {
            let n = results.len();
            queued.batch.results = std::mem::replace(&mut results, vec![None; n]);
            queued.batch.slots = slots;
            self.align_pending = Some(queued);
            return Ok(results);
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

    /// Finalises the aligning slots, or with the DTW queued, moves their state on and
    /// returns the alignment in flight (its `results` and `slots` left for the caller)
    fn align_and_finalize(
        &mut self,
        to_align: &[(usize, AlignItem)],
        results: &mut [Option<ChunkResult>],
    ) -> Result<Option<PendingAlign>> {
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
            match self.batched_dtw(to_align, &outs)? {
                Dtw::Done(texts) => Some(texts),
                Dtw::Queued(launch) => {
                    return self
                        .queue_finalize(to_align, &last_timings, launch)
                        .map(Some);
                }
            }
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

        Ok(None)
    }

    /// The deferred half of [`Self::align_and_finalize`]: each slot's state moves on
    /// now, and a helper thread waits for the matrices, runs the DTW and builds the texts
    fn queue_finalize(
        &mut self,
        to_align: &[(usize, AlignItem)],
        last_timings: &[f64],
        launch: Box<QueuedDtw>,
    ) -> Result<PendingAlign> {
        let mut plan = Vec::with_capacity(to_align.len());
        let mut entries = Vec::with_capacity(to_align.len());
        for (k, (orig, item)) in to_align.iter().enumerate() {
            let op = self.finalize_state(item.slot, last_timings[k], item.more, item.phase);
            // a finished chunk's text leaves the slot, which may take a new chunk before
            // the DTW lands; a continuing chunk's stays, and nothing touches it until
            // the texts are applied
            let sofar = matches!(op, TextOp::Finish { .. })
                .then(|| std::mem::take(&mut self.slots[item.slot].sofar));
            plan.push((op, sofar));
            entries.push((*orig, item.slot));
        }

        let QueuedDtw {
            mut matrices,
            prepared,
            offsets,
            texts,
            xa_slots,
        } = *launch;
        let score = self.config.align_override_score;
        let worker = std::thread::Builder::new()
            .name("dtw".to_owned())
            .spawn(move || -> Result<AlignedTexts> {
                let filled = prepared.align(matrices.wait()?)?;
                // the page-locked buffer goes back to the model now, not after the join
                drop(matrices);
                ensure!(
                    filled.len() == offsets.len() && plan.len() == offsets.len(),
                    "DTW returned {} texts for {} slots",
                    filled.len(),
                    offsets.len()
                );
                let outs = plan
                    .into_iter()
                    .zip(filled.iter().zip(&offsets))
                    .map(|((op, sofar), (text, &offset))| {
                        let cur = shift_window(text, offset)?;
                        Ok(match (sofar, op) {
                            (Some(mut sofar), op) => {
                                TextOut::Done(apply_text(&mut sofar, &cur, &op, score)?)
                            }
                            (None, TextOp::Continue { strip }) => {
                                TextOut::Append(window_piece(cur, strip)?)
                            }
                            (None, _) => TextOut::Done(None),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(AlignedTexts { filled, outs })
            })
            .wrap_err("start the DTW thread")?;

        Ok(PendingAlign {
            worker,
            batch: AlignBatch {
                results: Vec::new(),
                slots: Vec::new(),
                entries,
                texts,
                xa_slots,
            },
        })
    }

    /// Waits for the queued alignment, if any, and applies it: continuing slots get
    /// their window's text, and the batch's results join [`Transcriber::resolved`]
    pub(crate) fn resolve_align(&mut self) -> Result<()> {
        match self.align_pending.take() {
            Some(p) => {
                let joined = p.join()?;
                self.apply_align(joined)
            }
            None => Ok(()),
        }
    }

    /// Applies a joined alignment (see [`Self::resolve_align`])
    pub(crate) fn apply_align(&mut self, joined: JoinedAlign) -> Result<()> {
        let JoinedAlign {
            batch:
                AlignBatch {
                    mut results,
                    slots,
                    entries,
                    texts,
                    xa_slots,
                },
            texts: AlignedTexts { filled, outs },
        } = joined;
        emit!(
            self,
            Event::FillWordTimings {
                xa_slots: &xa_slots,
                texts: &texts,
                out: &filled,
            }
        );

        for ((orig, slot), out) in entries.into_iter().zip(outs) {
            match out {
                TextOut::Append(piece) => append_window(&mut self.slots[slot].sofar, &piece, true),
                TextOut::Done(r) => results[orig] = r,
            }
        }

        emit!(
            self,
            Event::BatchAlign {
                slots: &slots,
                results: &results,
            }
        );
        self.resolved.extend(results.into_iter().flatten());
        Ok(())
    }

    /// [`Self::resolve_align`] if the queued alignment's thread is done, without waiting
    pub(crate) fn resolve_align_if_done(&mut self) -> Result<()> {
        if self
            .align_pending
            .as_ref()
            .is_some_and(PendingAlign::finished)
        {
            self.resolve_align()?;
        }

        Ok(())
    }

    /// Whether the queued alignment, if any, will write slot `slot`'s text
    pub(crate) fn align_pending_on(&self, slot: usize) -> bool {
        self.align_pending
            .as_ref()
            .is_some_and(|p| p.batch.entries.iter().any(|&(_, s)| s == slot))
    }

    /// Settles any queued alignment and hands its finished chunks to the crop planner
    /// and `completed`
    pub(crate) fn settle_align(&mut self, completed: &mut Vec<ChunkResult>) -> Result<()> {
        self.resolve_align()?;
        self.hand_over_resolved(completed)
    }

    /// [`Self::settle_align`] only if that needs no wait: the alignment's thread is done,
    /// or nothing is queued
    pub(crate) fn settle_align_if_done(&mut self, completed: &mut Vec<ChunkResult>) -> Result<()> {
        self.resolve_align_if_done()?;
        self.hand_over_resolved(completed)
    }

    fn hand_over_resolved(&mut self, completed: &mut Vec<ChunkResult>) -> Result<()> {
        for r in std::mem::take(&mut self.resolved) {
            self.plan_crops(&r)?;
            completed.push(r);
        }

        Ok(())
    }

    /// The `fill_word_timings_batch` call: window-relative texts in, texts with
    /// DTW word times out, shifted by each slot's window offset. Done in place when the
    /// model's matrices are ready, otherwise queued for a helper thread
    fn batched_dtw(&mut self, to_align: &[(usize, AlignItem)], outs: &[String]) -> Result<Dtw> {
        let max_audio_length = self.config.max_audio_length;
        let mut slot_data: Vec<(String, i64)> = Vec::with_capacity(to_align.len());
        let mut xa_slots = Vec::with_capacity(to_align.len());
        let mut offsets = Vec::with_capacity(to_align.len());
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
            offsets.push(s.state.offset);
        }

        let tables = self.tables.clone();
        let tokenizer = TokenizerDtw::new(&tables.tok)?;
        // dtw_prep nests inside, so idle time under this range is the host DTW
        let dtw = trace::Range::new("dtw_cpu");
        let prepared = PreparedBatch::new(&slot_data, &tokenizer, &DtwOptions::default())?;
        let mut matrices = match prepared.batch() {
            Some(batch) => {
                let rows: Vec<usize> = batch.slots.iter().map(|&si| xa_slots[si]).collect();
                Some(self.model.dtw_launch(batch, &rows)?)
            }
            None => None,
        };
        if let Some(m) = matrices.take_if(|m| !m.ready()) {
            return Ok(Dtw::Queued(Box::new(QueuedDtw {
                matrices: m,
                prepared,
                offsets,
                texts: slot_data,
                xa_slots,
            })));
        }

        let filled = match matrices.as_mut() {
            Some(m) => prepared.align(m.wait()?)?,
            None => prepared.unchanged(),
        };
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

        filled
            .iter()
            .zip(&offsets)
            .map(|(text, &offset)| shift_window(text, offset))
            .collect::<Result<_>>()
            .map(Dtw::Done)
    }

    /// The window's text without a batched DTW result: finalised window-relative,
    /// then shifted by the window offset (DTW off, so `_dtw_fill_cur` is the identity)
    fn window_text(&self, slot_idx: usize, out: &str) -> Result<String> {
        let s = &self.slots[slot_idx];
        let cur = finalize_transcription(&finalize_state(s), out, false)?;
        shift_window(&cur, s.state.offset)
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
        // the window's text comes from the slot as it was before finalising, and only
        // a window that joins the chunk's text needs it
        let joins = phase == Phase::Process || self.slots[slot_idx].status() == Status::Finished;
        let cur = match pre_augmented {
            Some(c) => c.to_owned(),
            None if joins => self.window_text(slot_idx, out)?,
            None => String::new(),
        };
        let op = self.finalize_state(slot_idx, last_timing, more, phase);
        apply_text(
            &mut self.slots[slot_idx].sofar,
            &cur,
            &op,
            self.config.align_override_score,
        )
    }

    /// The part of `_finalize_slot_after_alignment` that doesn't read the window's
    /// text: the slot's state moves on (a continuing window restarts the slot at the
    /// advanced offset, a finished chunk leaves it ready), and the text work left is
    /// returned
    fn finalize_state(
        &mut self,
        slot_idx: usize,
        last_timing: f64,
        more: bool,
        phase: Phase,
    ) -> TextOp {
        let lang_set = self.config.lang.is_some();
        let speaker_tags = self.config.decode.speaker_tags;
        let init_len = self.init_tokens.len();
        let max_audio_length = self.config.max_audio_length;

        if phase == Phase::Process {
            let advance = if more {
                py::max(last_timing, MIN_ADVANCE)
            } else {
                last_timing
            };
            let last_token = *self.init_tokens.last().expect("non-empty start sequence");
            let s = &mut self.slots[slot_idx];
            s.state.offset += advance;
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
                return TextOp::Continue { strip: more };
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
            return TextOp::Skip;
        }

        let append = match phase {
            Phase::Process => Append::Process { strip: more },
            Phase::Finished => Append::Finished,
        };
        let s = &mut self.slots[slot_idx];
        if s.state.stop_reason.as_deref().is_none_or(str::is_empty) {
            s.state.stop_reason = Some("completed".to_owned());
        }

        s.state.status = Status::Ready;
        TextOp::Finish {
            append,
            meta: ResultMeta {
                id: s.id.clone().unwrap_or_default(),
                language: s.state.language.clone(),
                stop_reason: s.state.stop_reason.clone().unwrap_or_default(),
                offset: s.meta.offset,
                duration: s.meta.duration,
            },
        }
    }
}

/// A window's text shifted by the window offset onto the chunk's timeline
fn shift_window(text: &str, offset: f64) -> Result<String> {
    if offset > 0.0 {
        offset_timestamps(text, offset)
    } else {
        Ok(text.to_owned())
    }
}

/// What finalising a window does to the chunk's text once the window's text is known
#[derive(Debug, Clone)]
enum TextOp {
    /// A `finished` window on a slot that isn't finished: nothing
    Skip,
    /// The chunk continues: the window joins the slot's text
    Continue { strip: bool },
    /// The chunk ends: the window joins the text, which is cleaned into the result
    Finish { append: Append, meta: ResultMeta },
}

/// How a window joins the chunk's text
#[derive(Debug, Clone, Copy)]
enum Append {
    /// `process` phase, cut after the last closing timestamp when the chunk continues
    Process { strip: bool },
    /// `finished` phase, as is
    Finished,
}

/// The result fields taken from the slot when its chunk finishes
#[derive(Debug, Clone)]
struct ResultMeta {
    id: String,
    language: String,
    stop_reason: String,
    offset: Option<f64>,
    duration: Option<f64>,
}

/// A `process` window's text as it joins the chunk: when the chunk continues, the
/// next window re-covers anything after the last closing timestamp
fn window_piece(cur: String, strip: bool) -> Result<String> {
    if strip {
        strip_to_last_closing(&cur)
    } else {
        Ok(cur)
    }
}

/// The text half of `_finalize_slot_after_alignment`: appends the window `cur` to the
/// chunk's text `sofar` and, when the chunk ends, cleans it into the result
fn apply_text(
    sofar: &mut String,
    cur: &str,
    op: &TextOp,
    align_override_score: i64,
) -> Result<Option<ChunkResult>> {
    let (append, meta) = match op {
        TextOp::Skip => return Ok(None),
        TextOp::Continue { strip } => (Append::Process { strip: *strip }, None),
        TextOp::Finish { append, meta } => (*append, Some(meta)),
    };
    match append {
        Append::Process { strip } => {
            append_window(sofar, &window_piece(cur.to_owned(), strip)?, true);
        }
        Append::Finished => append_window(sofar, cur, false),
    }

    let Some(meta) = meta else {
        return Ok(None);
    };
    let cleaned = clean_chunk(sofar, align_override_score)?;
    let chunk_offset = meta.offset.unwrap_or(0.0);
    let prediction = if chunk_offset > 0.0 {
        offset_timestamps(&cleaned.text, chunk_offset)?
    } else {
        cleaned.text
    };
    Ok(Some(ChunkResult {
        id: meta.id.clone(),
        prediction,
        raw_cur: Some(sofar.clone()),
        language: meta.language.clone(),
        stop_reason: meta.stop_reason.clone(),
        speaker_id_map: Some(cleaned.speaker_order),
        offset: meta.offset,
        duration: meta.duration,
    }))
}

/// [`Transcriber::batched_dtw`]'s outcome
enum Dtw {
    /// Each slot's aligned text, shifted onto its chunk's timeline
    Done(Vec<String>),
    /// The model queued the GPU prep
    Queued(Box<QueuedDtw>),
}

/// A DTW whose matrices are in flight, with what the host half needs
struct QueuedDtw {
    matrices: Box<dyn DtwMatrices>,
    prepared: PreparedBatch,
    /// Each slot's window offset
    offsets: Vec<f64>,
    /// `(window_text, num_frames)` per slot, for the probe
    texts: Vec<(String, i64)>,
    xa_slots: Vec<usize>,
}

/// A text the DTW thread built for one aligning slot
enum TextOut {
    /// A continuing chunk's window, to append to the slot's text
    Append(String),
    /// A finished chunk's result, or `None`
    Done(Option<ChunkResult>),
}

/// What the DTW thread returns
struct AlignedTexts {
    /// `fill_word_timings_batch`'s texts, for the probe
    filled: Vec<String>,
    /// One per aligning slot, in order
    outs: Vec<TextOut>,
}

/// An alignment batch whose DTW runs on a helper thread
pub(crate) struct PendingAlign {
    worker: JoinHandle<Result<AlignedTexts>>,
    batch: AlignBatch,
}

/// What applying a queued alignment needs besides the thread's texts
struct AlignBatch {
    /// One per item, the skipped slots' already filled in
    results: Vec<Option<ChunkResult>>,
    /// Each item's slot
    slots: Vec<usize>,
    /// `(item index, slot)` of each aligning slot
    entries: Vec<(usize, usize)>,
    texts: Vec<(String, i64)>,
    xa_slots: Vec<usize>,
}

impl PendingAlign {
    /// Whether the DTW thread is done, so [`PendingAlign::join`] won't wait
    pub(crate) fn finished(&self) -> bool {
        self.worker.is_finished()
    }

    /// Waits for the DTW thread
    pub(crate) fn join(self) -> Result<JoinedAlign> {
        let _range = trace::Range::new("dtw_wait");
        let Self { worker, batch } = self;
        let texts = worker
            .join()
            .map_err(|_| eyre!("the DTW thread panicked"))??;
        Ok(JoinedAlign { batch, texts })
    }
}

/// A [`PendingAlign`] whose thread finished
pub(crate) struct JoinedAlign {
    batch: AlignBatch,
    texts: AlignedTexts,
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
