//! Constrained decoding: the per-slot state, the precomputed token tables and the
//! state machine that turns one logits row into the next token
//!
//! Ported from `decoding.py` (`DecoderState`, `decode_batch`,
//! `update_state_after_decode`). The CPU implementation here works one row at a time
//! and reproduces the GPU's fp16 mask, penalty and argmax arithmetic exactly

pub mod fsm;
pub mod logits;
pub mod mask;
pub mod state;
pub mod tables;
pub mod text_rules;

pub use fsm::{Decision, DecodeConfig, Decoder, Prepared, Sample, Stage};
pub use mask::TokenMask;
pub use state::{SlotState, Status};
pub use tables::DecoderTables;
