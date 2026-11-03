//! Transcript post-processing: every text pass between a window's raw decoder text
//! and the final `text` and `segments`
//!
//! Each function ports one Python function and keeps its name. The reference is the
//! Python pipeline's output byte for byte, so the ports reproduce CPython's string,
//! regex and float semantics ([`py`], [`pattern`]) rather than Rust's. Functions
//! return `Err` where the Python raises

pub mod finalize;
pub mod pattern;
pub mod py;
mod pyclass;
pub mod segments;
pub mod slot;
pub mod speakers;
pub mod tags;
pub mod timings;
pub mod transcript;

pub use finalize::{FinalizeState, finalize_transcription};
pub use segments::{Segment, Speaker, parse_to_segments, strip_to_last_closing};
pub use slot::{CleanedChunk, append_window, clean_chunk};
pub use speakers::{
    fix_speaker_ids, merge_consecutive_speakers, normalize_speaker_ids, remove_speaker_tags_not_after_newline,
};
pub use tags::{consolidate_adjacent_tags, remove_orphan_paired_tags};
pub use timings::{
    cut_after_last_timed_word, fix_nonmonotonic_timestamps, offset_timestamps, parse_timing, quantize_timestamp,
    quantize_timestamps, reconcile_decoder_and_alignment_timings, remove_invalid_timing_pairs,
    remove_short_nonspeech, remove_short_pauses, round_to,
};
pub use transcript::{Transcript, finish, finish_text, join_predictions};
