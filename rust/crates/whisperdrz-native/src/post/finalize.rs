//! `inference_utils.finalize_transcription`: the first pass over a window's raw
//! decoder text

use eyre::Result;

use super::pattern::{Lazy, PyRe};
use super::speakers::BRACKET_DIGIT;
use super::timings::quantize_timestamps;
use super::py;

static TAG_WITHOUT_NEWLINE: Lazy = Lazy::new(|| PyRe::fancy(r"(?<!\n)(?<=.)(\[\d\])"));
static TAG_AT_START: Lazy = Lazy::new(|| PyRe::plain(r"^\[\d\]"));

/// The `InferenceState` fields `finalize_transcription` reads
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FinalizeState {
    /// `state.offset`: the window's start in the chunk, in seconds
    pub offset: f64,
    /// `state.keepspeaker` (a class attribute that is always `False` in the pipeline)
    pub keepspeaker: bool,
    /// `state.cutearly`: drop the leading speaker tag when the window continues the
    /// previous speaker
    pub cutearly: bool,
}

/// `finalize_transcription`: quantizes markers (adding `state.offset` when
/// `apply_offset`), puts every single-digit speaker tag on its own line, ends the
/// text with a newline unless `keepspeaker`, and with `cutearly` drops a speaker tag
/// within the first ten characters
pub fn finalize_transcription(state: &FinalizeState, cur: &str, apply_offset: bool) -> Result<String> {
    let offset = if apply_offset { state.offset } else { 0.0 };
    let cur = quantize_timestamps(cur, offset, 0.02)?;
    let cur = cur.replace("[]", "\n[0]");
    let mut cur = TAG_WITHOUT_NEWLINE.sub_with(&cur, |m| format!("\n{}", m.group(1)))?;
    if TAG_AT_START.is_match(&cur)? {
        cur.insert(0, '\n');
    }

    if !state.keepspeaker {
        cur.push('\n');
    }

    if state.cutearly {
        let prefix = py::prefix_chars(&cur, 10);
        if let Some(m) = BRACKET_DIGIT.search(prefix)? {
            cur = py::lstrip(&cur[m.end()..]).to_owned();
        }
    }

    Ok(cur)
}
