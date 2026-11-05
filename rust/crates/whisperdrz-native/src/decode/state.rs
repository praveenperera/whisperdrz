//! The per-slot decoding state: the `InferenceState` fields (`inference_utils.py:218`)
//! that `decode_batch` and `update_state_after_decode` read or write

use eyre::{Result, bail, eyre};

use super::text_rules::{get_line, line_has_content};
use crate::text::{Token, py};

/// `InferenceState.status`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Ready,
    Warming,
    Started,
    Process,
    Finished,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Warming => "warming",
            Self::Started => "started",
            Self::Process => "process",
            Self::Finished => "finished",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "ready" => Self::Ready,
            "warming" => Self::Warming,
            "started" => Self::Started,
            "process" => Self::Process,
            "finished" => Self::Finished,
            other => bail!("unknown slot status {other:?}"),
        })
    }
}

/// Decoding state of one slot
#[derive(Debug, Clone, PartialEq)]
pub struct SlotState {
    /// Language code; `"auto"` maps to the English mask
    pub language: String,
    pub status: Status,
    /// Start of the current window within the chunk, seconds
    pub offset: f64,
    /// Chunk audio length in seconds (python's `state.duration`, `audio.shape[-1] / 16000`)
    pub duration: f64,
    /// Text decoded in the current window
    pub cur: String,
    pub last_timing: Option<f64>,
    pub token_count: usize,
    pub last_token_id: Token,
    /// The last 10 token ids
    pub recent_tokens: Vec<Token>,
    pub forced_timing_before_newline: bool,
    pub force_timing_every_word: bool,
    pub force_timestamp_next: bool,
    pub predict_next: Option<String>,
    pub stop_reason: Option<String>,
    /// `_window_max_timing`: largest timing seen in the window, kept across lines
    pub window_max_timing: f64,
    /// `_timing_count`: timing tokens on the current line
    pub timing_count: usize,
    /// `_line_cache`: the line `timing_count` and `last_timing` were computed for
    pub line_cache: String,
}

/// How many recent token ids a slot keeps
const RECENT_TOKENS: usize = 10;

impl SlotState {
    /// `duration` as python's `if state.duration` sees it
    pub(crate) fn has_duration(&self) -> bool {
        self.duration != 0.0
    }

    fn stop(&mut self, reason: String) {
        self.status = Status::Process;
        self.stop_reason = Some(reason);
    }

    /// `update_state_after_decode(state, newtext, newtoken_id, skip_cur)`
    /// (`decoding.py:878`): appends the token and applies the end-of-window checks
    pub fn update_after_decode(
        &mut self,
        newtext: &str,
        token: Token,
        skip_cur: bool,
    ) -> Result<()> {
        self.last_token_id = token;
        self.token_count += 1;
        self.recent_tokens.push(token);
        if self.recent_tokens.len() > RECENT_TOKENS {
            self.recent_tokens
                .drain(..self.recent_tokens.len() - RECENT_TOKENS);
        }

        // language and task tokens never reach the text
        let is_special =
            (newtext.starts_with("<|") && newtext.ends_with("|>") && !newtext.contains('.'))
                || newtext.is_empty();
        if !skip_cur && !is_special {
            self.cur.push_str(newtext);
        }

        let line = get_line(&self.cur).to_owned();
        let is_new_timing =
            newtext.starts_with("<|") && newtext.ends_with("|>") && newtext.contains('.');
        if newtext.contains('\n') {
            // stop when the line that just ended had timestamps but no words or tags
            if self.timing_count > 0 {
                let end = self
                    .cur
                    .rfind('\n')
                    .ok_or_else(|| eyre!("ValueError: substring not found"))?;
                if !line_has_content(&self.cur[..end]) {
                    self.stop("empty_line".to_owned());
                }
            }

            self.timing_count = 0;
            self.line_cache = line;
            self.last_timing = None;
        } else if is_new_timing {
            // python's newtext[2:-2], by code points
            let n = py::len(newtext);
            let inner: String = newtext.chars().skip(2).take(n.saturating_sub(4)).collect();
            match py::float(&inner) {
                Some(value) => {
                    self.timing_count += 1;
                    self.last_timing = Some(value);
                    // python's max() keeps the first argument unless the second is larger
                    if value > self.window_max_timing {
                        self.window_max_timing = value;
                    }

                    self.line_cache = line;
                }
                None => self.line_cache.clear(),
            }
        } else {
            self.line_cache = line;
        }

        if let Some(last_timing) = self.last_timing
            && self.timing_count.is_multiple_of(2)
        {
            // timestamps are window-relative: compare with the audio left in this window, at most 30 s
            let remaining = if self.has_duration() {
                self.duration - self.offset
            } else {
                30.0
            };
            let max_time = if 30.0 < remaining { 30.0 } else { remaining };
            if last_timing >= max_time - 0.1 {
                self.stop(format!(
                    "timing_exceeded:{last_timing:.2}>={:.2}",
                    max_time - 0.1
                ));
            }
        }

        // after a newline last_timing is gone, so this window-level check catches
        // hallucination past the end of a final (padded) window
        if self.window_max_timing > 0.0 && self.has_duration() && self.timing_count == 0 {
            let window_audio = self.duration - self.offset;
            if window_audio < 30.0 && self.window_max_timing >= window_audio - 1.0 {
                self.stop(format!(
                    "audio_exhausted:{:.2}>={:.2}",
                    self.window_max_timing,
                    window_audio - 1.0
                ));
            }
        }

        if self.cur.contains("[cutoff]") {
            let time_remaining = match self.last_timing {
                Some(t) if self.has_duration() => Some(self.duration - self.offset - t),
                _ => None,
            };
            if time_remaining.is_some_and(|t| t > 2.0) {
                self.cur = self.cur.replace("[cutoff]", "");
            } else {
                self.stop("cutoff_tag".to_owned());
            }
        }

        Ok(())
    }
}
