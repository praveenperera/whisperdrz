//! `_run_batched_inference` and the slot lifecycle around it: `_init_slot`,
//! `_prefill_slots`, the warm pool and the decode step

use std::sync::Arc;

use eyre::{Result, ensure, eyre};

use super::align::ChunkResult;
use super::graphs::DecodeStep;
use super::model::{Logits, MelBatch, Model, N_FRAMES, StepLogits};
use super::probe::Event;
use super::pyset::PySet;
use super::rows::Pending;
use super::slot::{Chunk, Slot};
use super::trace;
use super::{Transcriber, build_init_tokens};
use crate::decode::{Decision, Decoder, Prepared, Status};
use crate::text::Token;

/// Mel frames per second (`FRAMES_PER_SECOND`)
const FRAMES_PER_SECOND: f64 = 100.0;

/// Continuous-batching fallback: refill when at least this many slots are empty
const MIN_REFILL_BATCH: usize = 1;

/// `slots`' entries of `all`, in order, each listed once
fn rows_of<'s>(all: &'s mut [Slot], slots: &[usize]) -> Result<Vec<(usize, &'s mut Slot)>> {
    let mut by_slot: Vec<Option<&mut Slot>> = all.iter_mut().map(Some).collect();
    slots
        .iter()
        .map(|&s| {
            by_slot
                .get_mut(s)
                .and_then(Option::take)
                .map(|slot| (s, slot))
                .ok_or_else(|| eyre!("slot {s} missing or listed twice"))
        })
        .collect()
}

/// [`Decoder::prepare`] on each row, in order: the plan and mask, which read no logits
fn prepare_rows(
    decoder: &Decoder,
    rows: &mut [(usize, &mut Slot)],
    all_tokens: &[Vec<Token>],
) -> Result<Vec<Prepared>> {
    let _range = trace::Range::new("prepare");
    rows.iter_mut()
        .map(|(slot_idx, slot)| {
            let state = &mut slot.state;
            let window = Decoder::repetition_window(state.token_count)
                .map(|range| &all_tokens[*slot_idx][range]);
            decoder.prepare(state, window)
        })
        .collect()
}

/// The prepared rows' argmaxes and corrections over `logits` as they land
fn finish_rows(
    decoder: &Decoder,
    rows: Vec<(usize, &mut Slot)>,
    prepared: Vec<Prepared>,
    logits: &dyn StepLogits,
) -> Result<Vec<Decision>> {
    let _range = trace::Range::new("rows");
    ensure!(
        prepared.len() == rows.len(),
        "{} rows prepared of {}",
        prepared.len(),
        rows.len()
    );
    let pending = rows
        .into_iter()
        .zip(prepared)
        .map(|((_, slot), prepared)| Pending {
            state: &mut slot.state,
            prepared,
        })
        .collect();
    super::rows::finish(decoder, pending, logits)
}

impl<M: Model> Transcriber<M> {
    fn is_active(&self, slot: usize) -> bool {
        // `Finished` lasts only until the same step's alignment prep moves the slot to
        // `Ready`, unless a deferred alignment's error cut the step short; the retry must
        // still align it, or `flush` would spin on a slot nothing schedules
        matches!(
            self.slots[slot].status(),
            Status::Started | Status::Process | Status::Finished
        )
    }

    /// `_init_slot(slot_idx, chunk)`: resets the slot, computes the chunk's mel once
    /// and writes the start sequence
    pub(crate) fn init_slot(&mut self, slot_idx: usize, chunk: Chunk) -> Result<()> {
        let mel = (self.mel)(&chunk.audio)?;
        let speaker_tags = self.config.decode.speaker_tags;
        let max_audio_length = self.config.max_audio_length;

        let (init_tokens, effective_lang) = match &chunk.lang {
            Some(lang) => (
                build_init_tokens(&self.tables, Some(lang), speaker_tags)?,
                Some(lang.clone()),
            ),
            None => (self.init_tokens.clone(), self.config.lang.clone()),
        };
        ensure!(
            init_tokens.len() <= self.config.max_tokens,
            "start sequence longer than max_tokens"
        );

        let s = &mut self.slots[slot_idx];
        s.reset();
        s.continuation_count = 0;
        s.id = Some(chunk.id.clone());
        s.audio = Some(chunk.audio.clone());
        s.mel = Some(Arc::new(mel));
        let duration = s.duration().unwrap_or(0.0);
        s.state.duration = duration;

        s.state.status = Status::Started;
        s.state.timing_count = 0;
        s.state.line_cache.clear();
        s.state.last_timing = None;
        s.state.window_max_timing = 0.0;
        s.state.forced_timing_before_newline = false;
        s.state.predict_next = None;

        // `s.duration - s.offset if s.duration else 0`
        let remaining = if duration != 0.0 {
            duration - s.state.offset
        } else {
            0.0
        };
        s.is_final_chunk = remaining <= max_audio_length;

        // every key but id, audio and mel lands in meta; missing keys keep old values
        if chunk.offset.is_some() {
            s.meta.offset = chunk.offset;
        }

        if chunk.duration.is_some() {
            s.meta.duration = chunk.duration;
        }

        if let Some(lang) = &chunk.lang {
            s.state.language = lang.clone();
        }

        let row = &mut self.all_tokens[slot_idx];
        row.fill(0);
        row[..init_tokens.len()].copy_from_slice(&init_tokens);
        self.seq_lens[slot_idx] = init_tokens.len();
        s.state.token_count = init_tokens.len();
        s.state.last_token_id = *init_tokens
            .last()
            .ok_or_else(|| eyre!("empty start sequence"))?;

        if effective_lang.is_some() {
            s.state.cur = if speaker_tags {
                "[0] ".to_owned()
            } else {
                String::new()
            };
        }

        emit!(
            self,
            Event::InitSlot {
                slot: &self.slots[slot_idx],
                chunk: &chunk,
            }
        );
        Ok(())
    }

    /// `_kick_off_async_prefill(slots)`: prefills on the side stream and parks the
    /// slots as `warming` until their event completes
    fn kick_off_async_prefill(&mut self, slots: &[usize]) -> Result<()> {
        if slots.is_empty() {
            return Ok(());
        }

        for &s in slots {
            self.slots[s].state.status = Status::Warming;
        }

        self.model.begin_async_prefill()?;
        self.prefill_slots(slots)?;
        let event = self.model.end_async_prefill()?;
        for &s in slots {
            // the first-token decode may have stopped the slot; python overwrites that
            self.slots[s].state.status = Status::Warming;
            match self.prefill_events.iter_mut().find(|(k, _)| *k == s) {
                Some(entry) => entry.1 = event,
                None => self.prefill_events.push((s, event)),
            }
        }

        Ok(())
    }

    /// `_promote_warm_slots()`: slots whose prefill finished go back to `started`,
    /// the main stream waiting once per event
    fn promote_warm_slots(&mut self) -> Result<Vec<usize>> {
        let mut promoted = Vec::new();
        let mut synced = Vec::new();
        for (slot, event) in self.prefill_events.clone() {
            if self.model.prefill_done(event)? {
                if !synced.contains(&event) {
                    self.model.wait_prefill(event)?;
                    synced.push(event);
                }

                self.slots[slot].state.status = Status::Started;
                self.prefill_events.retain(|(k, _)| *k != slot);
                promoted.push(slot);
            }
        }

        Ok(promoted)
    }

    /// `_fill_warm_slots()`: keeps up to `n_warm` slots prefilled in the background,
    /// taking ready slots in index order
    fn fill_warm_slots(&mut self) -> Result<()> {
        if self.config.n_warm == 0 || self.pending.is_empty() {
            return Ok(());
        }

        let warming = self
            .slots
            .iter()
            .filter(|s| s.status() == Status::Warming)
            .count();
        let mut room = self.config.n_warm.saturating_sub(warming);
        if room == 0 {
            return Ok(());
        }

        let mut new_slots = Vec::new();
        for slot_idx in 0..self.slots.len() {
            if room == 0 || self.pending.is_empty() {
                break;
            }

            if self.slots[slot_idx].status() != Status::Ready {
                continue;
            }

            let chunk = self.pending.pop_front().expect("pending checked above");
            self.init_slot(slot_idx, chunk)?;
            new_slots.push(slot_idx);
            room -= 1;
        }

        self.kick_off_async_prefill(&new_slots)
    }

    /// `_prefill_slots(slots)`: resets the slots' KV, encodes each slot's current
    /// window of its mel, prefills the start sequence and decodes the first token
    pub(crate) fn prefill_slots(&mut self, slots: &[usize]) -> Result<()> {
        if slots.is_empty() {
            return Ok(());
        }

        self.model.reset_kv(slots)?;
        for &s in slots {
            self.kv_lens[s] = 0;
        }

        // every slot has a mel from `_init_slot`, so this is python's `has_mels` path
        let mut data = Vec::new();
        let mut n_mels = None;
        for &slot_idx in slots {
            let s = &mut self.slots[slot_idx];
            s.last_offset = s.state.offset;
            let mel = s
                .mel
                .clone()
                .ok_or_else(|| eyre!("slot {slot_idx} has no mel"))?;
            ensure!(
                *n_mels.get_or_insert(mel.n_mels) == mel.n_mels,
                "mels of different heights in one batch"
            );
            // `int(s.offset * 100)` truncates the product
            let start = (s.state.offset * FRAMES_PER_SECOND) as usize;
            mel.window_f16(start, &mut data);
        }

        let n_mels = n_mels.unwrap_or(0);
        debug_assert_eq!(data.len(), slots.len() * n_mels * N_FRAMES);
        self.model.encode(
            &MelBatch {
                rows: slots.len(),
                n_mels,
                data,
            },
            slots,
        )?;

        let init_len = self.init_tokens.len();
        let mut tokens = Vec::with_capacity(slots.len() * init_len);
        for &s in slots {
            tokens.extend_from_slice(&self.all_tokens[s][..init_len]);
        }

        let logits = self.model.prefill(&tokens, slots)?;
        for &s in slots {
            self.kv_lens[s] = init_len;
        }

        self.decode_and_update(&logits, slots, false)?;
        emit!(self, Event::PrefillSlots { slots });
        Ok(())
    }

    /// `decode_batch` over `slots` (row `i` of `logits` is `slots[i]`), then
    /// `update_state_after_decode` per row and the token writes. `step` marks the
    /// main decode loop, which also applies the non-English early exit
    fn decode_and_update(&mut self, logits: &Logits, slots: &[usize], step: bool) -> Result<()> {
        let _range = trace::Range::new("host_decode");
        self.emit_decode_before(slots);
        let tables = self.tables.clone();
        let decoder = Decoder::new(&tables, self.config.decode)?;
        let mut rows = rows_of(&mut self.slots, slots)?;
        let prepared = prepare_rows(&decoder, &mut rows, &self.all_tokens)?;
        let decisions = finish_rows(&decoder, rows, prepared, logits)?;
        self.update_after_decode(slots, &decisions, step)
    }

    fn emit_decode_before(&mut self, slots: &[usize]) {
        emit!(
            self,
            Event::DecodeBefore {
                slots,
                all: &self.slots,
                tokens: &self.all_tokens,
            }
        );
    }

    /// The decisions' `update_state_after_decode` and token writes, in row order
    fn update_after_decode(
        &mut self,
        slots: &[usize],
        decisions: &[Decision],
        step: bool,
    ) -> Result<()> {
        emit!(
            self,
            Event::DecodeAfter {
                slots,
                decisions,
                all: &self.slots,
            }
        );

        let max_seq_len = self.config.max_tokens;
        let detect = self.config.lang.is_none();
        for (&slot_idx, d) in slots.iter().zip(decisions) {
            let s = &mut self.slots[slot_idx];
            s.state.update_after_decode(&d.text, d.token, false)?;
            let new_len = s.state.token_count;
            if new_len <= max_seq_len {
                self.all_tokens[slot_idx][new_len - 1] = d.token;
                self.seq_lens[slot_idx] = new_len;
            }

            if step
                && self.config.skip_non_english
                && detect
                && !s.state.language.is_empty()
                && s.state.language != "en"
                && !matches!(s.state.status, Status::Finished | Status::Ready)
            {
                // a queued window's text lands before the clear, as it would have had
                // the alignment joined before this step's updates
                if self.align_pending_on(slot_idx) {
                    self.resolve_align()?;
                }

                let s = &mut self.slots[slot_idx];
                s.state.status = Status::Finished;
                s.state.stop_reason = Some(format!("skip_non_english:{}", s.state.language));
                s.sofar.clear();
                s.state.cur.clear();
            }

            emit!(
                self,
                Event::UpdateState {
                    slot: &self.slots[slot_idx],
                    text: &d.text,
                    token: d.token,
                }
            );
        }

        Ok(())
    }

    /// One decode step over `decode_slots`: their last tokens at their KV
    /// positions, padded to a bucket. The rows are planned and masked while the GPU
    /// runs the step, since neither reads its logits, and finished as the logits land
    fn decode_step(&mut self, decode_slots: &[usize]) -> Result<()> {
        let mut tokens = Vec::with_capacity(decode_slots.len());
        let mut positions = Vec::with_capacity(decode_slots.len());
        for &s in decode_slots {
            let seq_len = self.seq_lens[s];
            ensure!(seq_len > 0, "slot {s} has no tokens");
            tokens.push(self.all_tokens[s][seq_len - 1]);
            positions.push(i32::try_from(self.kv_lens[s])?);
        }

        let trash = self.slots.len();
        let step = DecodeStep::new(
            tokens,
            positions,
            decode_slots.to_vec(),
            &self.buckets,
            trash,
        );
        if step.is_padded() {
            self.kv_lens[trash] = 0;
        }

        let host = trace::Range::new("host_decode");
        let tables = self.tables.clone();
        let decoder = Decoder::new(&tables, self.config.decode)?;
        let all_tokens = &self.all_tokens;
        let probe = &mut self.probe;
        // moved into `rows` inside the callback, after the step is queued: the probe's
        // before-decode event follows the model call, as python's decode_batch does
        let mut all = Some(&mut self.slots[..]);
        let mut rows = Vec::new();
        let mut prepared = Vec::new();
        let logits = self.model.launch_step(&step, &mut || {
            let all = all.take().ok_or_else(|| eyre!("step callback ran twice"))?;
            if let Some(probe) = probe.as_mut() {
                probe(&Event::DecodeBefore {
                    slots: decode_slots,
                    all,
                    tokens: all_tokens,
                });
            }

            rows = rows_of(all, decode_slots)?;
            prepared = prepare_rows(&decoder, &mut rows, all_tokens)?;
            Ok(())
        })?;
        let decisions = finish_rows(&decoder, rows, prepared, &*logits)?;
        // the step's GPU errors surface before its state changes, as with the old
        // synchronous download
        logits.complete()?;
        drop(logits);
        drop(host);
        // the last alignment's forward runs on its own stream beside the steps, so its
        // texts go in once its thread is done, without a wait. Until then nothing reads
        // or writes the slots' texts: the next alignment settles it first, and the
        // non-English stop in `update_after_decode` waits for it before clearing one
        self.resolve_align_if_done()?;

        for &s in decode_slots {
            self.kv_lens[s] += 1;
        }

        let _range = trace::Range::new("host_update");
        self.update_after_decode(decode_slots, &decisions, true)
    }

    /// Takes the next pending chunk into `slot_idx`
    fn take_chunk(&mut self, slot_idx: usize) -> Result<()> {
        let chunk = self
            .pending
            .pop_front()
            .ok_or_else(|| eyre!("no pending chunk"))?;
        self.init_slot(slot_idx, chunk)
    }

    /// `_run_batched_inference()`: fills empty slots, then decodes with continuous
    /// batching for up to `2 * max_tokens` steps, aligning and refilling slots as
    /// windows end. Returns the chunks finished in this call
    pub(crate) fn run_batched_inference(&mut self) -> Result<Option<Vec<ChunkResult>>> {
        let n_slots = self.slots.len();
        let mut empty =
            PySet::from_items((0..n_slots).filter(|&i| self.slots[i].status() == Status::Ready));
        // slots may still be in "process" from a previous call
        let mut active: Vec<usize> = (0..n_slots).filter(|&i| self.is_active(i)).collect();
        let mut completed = Vec::new();

        let mut new_slots = Vec::new();
        for slot_idx in empty.to_vec() {
            if self.pending.is_empty() {
                break;
            }

            self.take_chunk(slot_idx)?;
            new_slots.push(slot_idx);
            empty.discard(slot_idx);
            active.push(slot_idx);
        }

        self.prefill_slots(&new_slots)?;
        self.fill_warm_slots()?;
        if active.is_empty() {
            // an alignment an earlier call queued and didn't settle (it returned an
            // error) still owes its results
            let mut completed = Vec::new();
            self.settle_align(&mut completed)?;
            return Ok((!completed.is_empty()).then_some(completed));
        }

        let max_seq_len = self.config.max_tokens;
        for _step in 0..max_seq_len * 2 {
            let warm = trace::Range::new("warm");
            let promoted = self.promote_warm_slots()?;
            active.extend(promoted);
            active.retain(|&i| self.is_active(i));
            self.fill_warm_slots()?;
            drop(warm);

            // sync refill, only when nothing is decoding or warming
            if active.is_empty() && !self.pending.is_empty() && self.prefill_events.is_empty() {
                let empty_now: Vec<usize> = (0..n_slots)
                    .filter(|&i| self.slots[i].status() == Status::Ready)
                    .collect();
                let mut new_slots = Vec::new();
                for slot_idx in empty_now {
                    if self.pending.is_empty() {
                        break;
                    }

                    self.take_chunk(slot_idx)?;
                    new_slots.push(slot_idx);
                    active.push(slot_idx);
                }

                self.prefill_slots(&new_slots)?;
                self.fill_warm_slots()?;
            }

            if active.is_empty() && self.pending.is_empty() && self.prefill_events.is_empty() {
                break;
            }

            // safety stops: window token budget, and implausibly fast token rates
            for &slot_idx in &active {
                let s = &mut self.slots[slot_idx].state;
                if s.status != Status::Started {
                    continue;
                }

                let token_count = s.token_count;
                if token_count + 2 > max_seq_len {
                    s.status = Status::Process;
                    s.stop_reason = Some(format!("safety_max_tokens:{token_count}"));
                    continue;
                }

                if s.duration != 0.0 && token_count as f64 > 20.0 * s.duration {
                    s.status = Status::Process;
                    s.stop_reason = Some(format!(
                        "excessive_tokens:{token_count}>{:.0}",
                        20.0 * s.duration
                    ));
                }
            }

            let decode_slots: Vec<usize> = active
                .iter()
                .copied()
                .filter(|&i| self.slots[i].status() == Status::Started)
                .collect();
            if !decode_slots.is_empty() {
                self.decode_step(&decode_slots)?;
            }

            // a queued alignment lands before the next window ends: waited for when a
            // slot is about to align (its text reads the slot's), taken if done otherwise
            let aligning = active
                .iter()
                .any(|&i| matches!(self.slots[i].status(), Status::Process | Status::Finished));
            if aligning {
                self.settle_align(&mut completed)?;
            } else {
                self.settle_align_if_done(&mut completed)?;
            }

            let align_prep = trace::Range::new("align_prep");
            let mut pending_align = Vec::new();
            let mut finished = Vec::new();
            for slot_idx in active.clone() {
                if matches!(
                    self.slots[slot_idx].status(),
                    Status::Process | Status::Finished
                ) && let Some(item) = self.prepare_slot_for_alignment(slot_idx)?
                {
                    pending_align.push(item);
                    finished.push(slot_idx);
                }
            }

            drop(align_prep);
            if pending_align.is_empty() {
                continue;
            }

            let results = self.batch_align_slots(pending_align)?;
            let mut needing_continue = Vec::new();
            for (slot_idx, result) in finished.into_iter().zip(results) {
                if let Some(r) = result {
                    self.plan_crops(&r)?;
                    completed.push(r);
                }

                match self.slots[slot_idx].status() {
                    Status::Ready => {
                        self.model.reset_kv(&[slot_idx])?;
                        self.kv_lens[slot_idx] = 0;
                        if let Some(p) = active.iter().position(|&i| i == slot_idx) {
                            active.remove(p);
                        }

                        empty.add(slot_idx);
                    }
                    Status::Started => needing_continue.push(slot_idx),
                    _ => {}
                }
            }

            let should_refill =
                !self.pending.is_empty() && (empty.len() >= MIN_REFILL_BATCH || active.is_empty());
            if should_refill {
                let refill_count = empty.len().min(self.pending.len());
                let refill: Vec<usize> = empty.iter().take(refill_count).collect();
                for &slot_idx in &refill {
                    self.take_chunk(slot_idx)?;
                    empty.discard(slot_idx);
                    active.push(slot_idx);
                }

                needing_continue.extend(refill);
            }

            self.prefill_slots(&needing_continue)?;
        }

        self.settle_align(&mut completed)?;
        Ok((!completed.is_empty()).then_some(completed))
    }
}
