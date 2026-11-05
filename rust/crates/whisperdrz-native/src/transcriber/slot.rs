//! One transcriber slot: the `InferenceState` (`inference_utils.py:217`) fields the
//! scheduler reads and writes, around the decoder's [`SlotState`]

use std::sync::Arc;

use super::model::Mel;
use crate::decode::{SlotState, Status};
use crate::text::Token;

/// Audio sample rate
pub const SAMPLE_RATE: usize = 16_000;

/// A unit of work: one chunk of 16 kHz audio (the pipeline's chunk dict)
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    /// `chunk["id"]`; `transcribe` sorts results by the number after `_`
    pub id: String,
    pub audio: Arc<[f32]>,
    /// `chunk["offset"]`: added to the finished chunk's timestamps. A chunk without
    /// one keeps the slot's previous value, as the slot's `meta` dict does
    pub offset: Option<f64>,
    /// `chunk["duration"]`, kept in `meta` only (the slot's duration comes from the audio)
    pub duration: Option<f64>,
    /// `chunk["lang"]`: per-chunk language override
    pub lang: Option<String>,
}

/// The slot's `meta` dict, limited to the keys the scheduler reads or returns
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Meta {
    pub offset: Option<f64>,
    pub duration: Option<f64>,
}

/// One slot's state
#[derive(Debug, Clone)]
pub struct Slot {
    pub index: usize,
    pub id: Option<String>,
    /// The decoder's view: status, window offset, duration, `cur`, counters, flags
    pub state: SlotState,
    pub audio: Option<Arc<[f32]>>,
    /// The chunk's mel, computed once in `_init_slot` and sliced per window
    pub mel: Option<Arc<Mel>>,
    /// Text of the windows finished so far, each followed by `[CHOP]`
    pub sofar: String,
    /// Window offset of the last prefill (`s.last_offset`)
    pub last_offset: f64,
    pub cutearly: bool,
    pub continuation_count: u32,
    pub is_final_chunk: bool,
    pub meta: Meta,
    pub fails: u32,
}

impl Slot {
    /// A fresh `InferenceState(index=i, language=lang)`
    pub fn new(index: usize, language: &str) -> Self {
        Self {
            index,
            id: None,
            state: SlotState {
                language: language.to_owned(),
                status: Status::Ready,
                offset: 0.0,
                duration: 0.0,
                cur: String::new(),
                last_timing: None,
                token_count: 0,
                // python starts at -1; every path sets it before it is read
                last_token_id: Token::MAX,
                recent_tokens: Vec::new(),
                forced_timing_before_newline: false,
                force_timing_every_word: false,
                force_timestamp_next: false,
                predict_next: None,
                stop_reason: None,
                window_max_timing: 0.0,
                timing_count: 0,
                line_cache: String::new(),
            },
            audio: None,
            mel: None,
            sofar: String::new(),
            last_offset: -1.0,
            cutearly: false,
            continuation_count: 0,
            is_final_chunk: false,
            meta: Meta::default(),
            fails: 0,
        }
    }

    pub fn status(&self) -> Status {
        self.state.status
    }

    /// `InferenceState.reset()`. `language`, `force_timestamp_next`,
    /// `_timing_count`, `_line_cache` and `meta` survive it, as in python
    pub fn reset(&mut self) {
        let s = &mut self.state;
        s.last_timing = None;
        s.offset = 0.0;
        s.cur.clear();
        self.sofar.clear();
        s.last_token_id = Token::MAX;
        s.token_count = 0;
        s.recent_tokens.clear();
        self.fails = 0;
        self.cutearly = false;
        s.forced_timing_before_newline = false;
        s.predict_next = None;
        s.stop_reason = None;
        s.window_max_timing = 0.0;
    }

    /// `InferenceState.duration`: the audio length in seconds, or `meta["duration"]`
    /// without audio
    pub fn duration(&self) -> Option<f64> {
        match &self.audio {
            Some(a) => Some(a.len() as f64 / SAMPLE_RATE as f64),
            None => self.meta.duration,
        }
    }
}
