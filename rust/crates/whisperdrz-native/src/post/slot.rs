//! The text side of `Transcriber._finalize_slot_after_alignment`: appending a
//! window to the slot's running text and cleaning the finished chunk

use eyre::Result;

use super::pattern::{Lazy, PyRe};
use super::speakers::{
    BRACKET_DIGIT, fix_speaker_ids, merge_consecutive_speakers, remove_speaker_tags_not_after_newline,
};
use super::tags::{consolidate_adjacent_tags, remove_orphan_paired_tags};
use super::timings::{
    fix_nonmonotonic_timestamps, reconcile_decoder_and_alignment_timings, remove_invalid_timing_pairs,
    remove_short_nonspeech, remove_short_pauses,
};
use super::py;

static DANGLING_QUOTE_BRACKET: Lazy = Lazy::new(|| PyRe::fancy(r#"\s*\["\s*(?=\n|$)"#));
static DANGLING_BINARY_DIGIT: Lazy = Lazy::new(|| PyRe::fancy(r"\s+[01]\s*(?=\n|$)"));
static LEADING_CLOSE_BRACKET: Lazy = Lazy::new(|| PyRe::fancy(r"(?<=\n)]|^]"));
static TRAILING_OPEN_BRACKET: Lazy = Lazy::new(|| PyRe::fancy(r"\[(?=\n|$)"));
static LONE_BANG: Lazy = Lazy::new(|| PyRe::fancy(r"(?<![a-zA-Z])!(?![a-zA-Z])"));
// `\bCouncill`: the word boundary is checked by hand against CPython's `\w`
static COUNCILL: Lazy = Lazy::new(|| PyRe::plain(r"Councill(?:<\|[\d.]+\|>)?"));
static TAG_ONLY_LINE: Lazy = Lazy::new(|| {
    PyRe::plain(r"(?m)^\[(?:\d+|c)\]\s*(?:<\|[\d.]+\|>\s*)*(?:\[[a-zA-Z]+\]\s*(?:<\|[\d.]+\|>\s*)*)+\s*$")
});
static NEWLINE_RUN: Lazy = Lazy::new(|| PyRe::plain(r"\n{2,}"));
static ORPHAN_TIMESTAMP: Lazy = Lazy::new(|| PyRe::plain(r"(\[\d+\] )<\|[\d.]+\|> (<\|[\d.]+\|>)"));

/// Appends one window's text and a `[CHOP]` to the slot's running text (`s.sofar`).
/// In the `process` phase (`process_phase`) the window joins on a newline when it
/// starts with a tag, a space otherwise, and nothing when the text so far is empty
/// or ends in a newline; in the `finished` phase it is appended as is
pub fn append_window(sofar: &mut String, cur: &str, process_phase: bool) {
    if process_phase {
        let content = sofar.replace("[CHOP]", "");
        let content = py::rstrip(&content);
        let sep = if content.is_empty() || content.ends_with('\n') {
            ""
        } else if cur.starts_with('[') {
            "\n"
        } else {
            " "
        };
        sofar.push_str(sep);
    }

    sofar.push_str(cur);
    sofar.push_str("[CHOP]");
}

/// `re.sub(r"\bCouncill(?:<\|[\d.]+\|>)?", "", text)` with CPython's `\b`
fn remove_councill(text: &str) -> Result<String> {
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    let mut pos = 0;
    while let Some(m) = COUNCILL.search_from(text, pos)? {
        let boundary = text[..m.start()].chars().next_back().is_none_or(|c| !py::is_word(c));
        if boundary {
            out.push_str(&text[last..m.start()]);
            last = m.end();
            pos = m.end();
        } else {
            // `C` is one byte, so the next candidate starts one byte on
            pos = m.start() + 1;
        }
    }

    out.push_str(&text[last..]);
    Ok(out)
}

/// A finished chunk's cleaned text and the speaker ids it had before renumbering
#[derive(Debug, Clone, PartialEq)]
pub struct CleanedChunk {
    /// `meta["prediction"]` before the chunk offset is applied
    pub text: String,
    /// Original speaker id of each renumbered id (`speaker_id_map`)
    pub speaker_order: Vec<i64>,
}

/// The cleanup chain run on `s.sofar` when a chunk finishes; `align_override_score`
/// is `Transcriber.align_override_score`. The caller applies the chunk offset with
/// `offset_timestamps` when it is positive
pub fn clean_chunk(sofar: &str, align_override_score: i64) -> Result<CleanedChunk> {
    let c = reconcile_decoder_and_alignment_timings(sofar, align_override_score)?;
    let c = consolidate_adjacent_tags(&c)?;
    let c = remove_invalid_timing_pairs(&c, 0.02)?;
    let c = remove_speaker_tags_not_after_newline(&c)?;
    let c = merge_consecutive_speakers(&c)?;
    let c = fix_nonmonotonic_timestamps(&c)?;
    let c = DANGLING_QUOTE_BRACKET.sub(&c, "")?;
    let c = DANGLING_BINARY_DIGIT.sub(&c, "")?;
    let c = LEADING_CLOSE_BRACKET.sub(&c, "")?;
    let c = TRAILING_OPEN_BRACKET.sub(&c, "")?;
    let c = LONE_BANG.sub(&c, "")?;
    let c = remove_councill(&c)?;
    let c = remove_orphan_paired_tags(&c)?;
    let c = TAG_ONLY_LINE.sub(&c, "")?;
    let c = NEWLINE_RUN.sub(&c, "\n")?;
    let c = remove_short_nonspeech(&c, 0.02)?;
    let c = remove_short_pauses(&c, 800)?;
    let c = ORPHAN_TIMESTAMP.sub_with(&c, |m| format!("{}{}", m.group(1), m.group(2)))?;

    let mut speaker_order = Vec::new();
    for m in BRACKET_DIGIT.captures_all(&c)? {
        let id = py::int(m.group(1))?;
        if !speaker_order.contains(&id) {
            speaker_order.push(id);
        }
    }

    Ok(CleanedChunk { text: fix_speaker_ids(&c)?, speaker_order })
}
