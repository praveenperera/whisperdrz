//! Observation points in the scheduler, one per function the reference tracer
//! records, so a replay can compare every call with the Python trace

use super::align::{AlignItem, ChunkResult};
use super::slot::{Chunk, Slot};
use crate::decode::Decision;

/// One scheduler call, reported when it completes unless noted
#[derive(Debug)]
pub enum Event<'a> {
    /// `_init_slot(slot, chunk)`
    InitSlot { slot: &'a Slot, chunk: &'a Chunk },
    /// `_prefill_slots(slots)`
    PrefillSlots { slots: &'a [usize] },
    /// `decode_batch` is about to run on these slots (`slots[i]` is row `i`);
    /// `all` is every slot and `tokens` every slot's `all_tokens` row, indexed by
    /// slot number
    DecodeBefore {
        slots: &'a [usize],
        all: &'a [Slot],
        tokens: &'a [Vec<u32>],
    },
    /// `decode_batch` returned `decisions`, before the per-row state updates
    DecodeAfter {
        slots: &'a [usize],
        decisions: &'a [Decision],
        all: &'a [Slot],
    },
    /// `update_state_after_decode(state, text, token)` on one slot, state after
    UpdateState {
        slot: &'a Slot,
        text: &'a str,
        token: u32,
    },
    /// `_prepare_slot_for_alignment(s)` returned `item`
    PrepareAlignment {
        slot: &'a Slot,
        item: Option<&'a AlignItem>,
    },
    /// `fill_word_timings_batch(slot_data)`: `texts[i]` with `num_frames[i]` attends
    /// to `xa_cache[xa_slots[i]]`
    FillWordTimings {
        xa_slots: &'a [usize],
        texts: &'a [(String, i64)],
        out: &'a [String],
    },
    /// `_finalize_slot_after_alignment(...)`, slot state after
    Finalize {
        slot: &'a Slot,
        out: &'a str,
        last_timing: f64,
        more: bool,
        phase: &'a str,
        pre_augmented: Option<&'a str>,
        result: Option<&'a ChunkResult>,
    },
    /// `_batch_align_slots(items)` returned `results`
    BatchAlign {
        slots: &'a [usize],
        results: &'a [Option<ChunkResult>],
    },
}

/// Receives [`Event`]s
pub type Probe = Box<dyn FnMut(&Event<'_>)>;
