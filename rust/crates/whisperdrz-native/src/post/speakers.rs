//! `clean/speakers.py` and `normalize_speaker_ids`: speaker tag numbering and line
//! merging

use eyre::Result;

use super::pattern::{Lazy, PyRe};
use super::py;

/// `SPEAKER_LINE_START`: `[0]` or `[c]` at the start, capturing the id, plus the
/// whitespace after it
pub static SPEAKER_LINE_START: Lazy = Lazy::new(|| PyRe::plain(r"^\[(\d+|c)\]\s*"));
/// `BRACKET_DIGIT_MATCH`: `[0]`, capturing the digits
pub static BRACKET_DIGIT: Lazy = Lazy::new(|| PyRe::plain(r"\[(\d+)\]"));
static SPEAKER_TAGS_GENERAL: Lazy = Lazy::new(|| PyRe::fancy(r"(?<=[^\n])\[(?:\d+|c)\]\s*"));

/// Renumbers `[N]` tags by first appearance, keying on the digit string (so `[01]`
/// and `[1]` are different speakers)
fn renumber(text: &str) -> Result<String> {
    let mut seen: Vec<String> = Vec::new();
    BRACKET_DIGIT.sub_with(text, |m| {
        let id = m.group(1);
        let n = match seen.iter().position(|s| s == id) {
            Some(n) => n,
            None => {
                seen.push(id.to_owned());
                seen.len() - 1
            }
        };
        format!("[{n}]")
    })
}

/// `fix_speaker_ids`: makes `[N]` ids count up from 0 in order of appearance
pub fn fix_speaker_ids(text: &str) -> Result<String> {
    renumber(text)
}

/// `speaker_clustering.normalize_speaker_ids`: the same renumbering, applied to the
/// joined transcript
pub fn normalize_speaker_ids(text: &str) -> Result<String> {
    renumber(text)
}

/// `remove_speaker_tags_not_after_newline`: drops `[N]`/`[c]` tags (and the
/// whitespace after them) that don't start the text or a line
pub fn remove_speaker_tags_not_after_newline(text: &str) -> Result<String> {
    SPEAKER_TAGS_GENERAL.sub(text, "")
}

/// `merge_consecutive_speakers`: within each `[CHOP]` chunk, drops blank lines and
/// joins a line onto the previous one when both start with the same speaker tag
pub fn merge_consecutive_speakers(text: &str) -> Result<String> {
    let mut chunks = Vec::new();
    for chunk in text.split("[CHOP]") {
        let mut merged: Vec<String> = Vec::new();
        for line in chunk.split('\n') {
            if py::strip(line).is_empty() {
                continue;
            }

            if let (Some(m), Some(prev)) = (SPEAKER_LINE_START.search(line)?, merged.last_mut())
                && let Some(prev_m) = SPEAKER_LINE_START.search(prev)?
                && prev_m.group(1) == m.group(1)
            {
                let joined = format!("{} {}", py::rstrip(prev), &line[m.end()..]);
                *prev = joined;
                continue;
            }

            merged.push(line.to_owned());
        }

        chunks.push(merged.join("\n"));
    }

    Ok(chunks.join("[CHOP]"))
}
