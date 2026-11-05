//! The line-inspection helpers of `decoding.py` (lines 26-235, 2049-2157): word
//! characters, timed spans, open brackets and the valid timing range
//!
//! Positions are byte offsets. Python compares code-point offsets, which order the same
//! way, and every length rule counts code points through `py::len`

use std::sync::LazyLock;

use regex::Regex;

use super::tables::{MAX_TIMING_IDX, TIMING_STEP, parse_timing};
use crate::text::py;

/// `clean.TIMINGS_FORMAT`: `<|1.23|>`, capturing the number. `\d` is Unicode Nd in
/// both python's `re` and the `regex` crate
pub static TIMINGS_FORMAT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<\|(\d+\.\d+)\|>").unwrap());
static TIMED_SPAN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<\|\d+\.\d+\|>([^<]*)<\|\d+\.\d+\|>").unwrap());
static BRACKETED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[[^\]]*\]").unwrap());

/// `is_word_char`: alphanumeric, or a combining mark (Hindi/Thai/Tamil vowel signs)
pub fn is_word_char(c: Option<char>) -> bool {
    c.is_some_and(|c| py::is_alnum(c) || py::is_mark(c))
}

/// `text_has_word`
pub fn text_has_word(s: &str) -> bool {
    s.chars().any(|c| is_word_char(Some(c)))
}

/// `_get_line`: text after the last newline
pub fn get_line(cur: &str) -> &str {
    cur.rfind('\n').map_or(cur, |i| &cur[i + 1..])
}

/// `line.split("] ", 1)[-1] if "] " in line else line`
pub fn after_speaker(line: &str) -> &str {
    line.split_once("] ").map_or(line, |(_, rest)| rest)
}

/// `_line_has_content`: the current line has a word or a tag, not just timestamps
pub fn line_has_content(cur: &str) -> bool {
    let stripped = TIMINGS_FORMAT.replace_all(after_speaker(get_line(cur)), "");
    text_has_word(&stripped) || stripped.contains('[')
}

/// `_has_timed_word`: some `<|t|>content<|t|>` span has a word outside `[tags]`
pub fn has_timed_word(line: &str) -> bool {
    TIMED_SPAN
        .captures_iter(line)
        .any(|c| text_has_word(&BRACKETED.replace_all(&c[1], "")))
}

/// `_has_timed_content`: some `<|t|>content<|t|>` span is not blank
pub fn has_timed_content(line: &str) -> bool {
    TIMED_SPAN
        .captures_iter(line)
        .any(|c| !py::strip(&c[1]).is_empty())
}

/// Number of whitespace-separated words once `[ ] * ( )` are removed
/// (`_BRACKETS_CLEAN.sub("", s).split()`)
pub fn count_words(s: &str) -> usize {
    let cleaned: String = s
        .chars()
        .filter(|c| !matches!(c, '[' | ']' | '*' | '(' | ')'))
        .collect();
    py::split_whitespace(&cleaned).count()
}

/// Byte offsets of `<` not followed by `|` (`_TAG_OPEN`)
pub fn tag_opens(line: &str) -> Vec<usize> {
    let b = line.as_bytes();
    (0..b.len())
        .filter(|&i| b[i] == b'<' && b.get(i + 1) != Some(&b'|'))
        .collect()
}

/// Byte offsets of `>` not preceded by `|` (`_TAG_CLOSE`)
pub fn tag_closes(line: &str) -> Vec<usize> {
    let b = line.as_bytes();
    (0..b.len())
        .filter(|&i| b[i] == b'>' && (i == 0 || b[i - 1] != b'|'))
        .collect()
}

const ASTERISK_MAX_LEN: usize = 30;

/// Why an open `*...` span has to be closed now
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsteriskClose {
    SpaceInPartial,
    NewlineInPartial,
    TooLong,
}

/// `asterisk_state(line)`: (inside an unclosed `*`, reason to force the close)
pub fn asterisk_state(line: &str) -> (bool, Option<AsteriskClose>) {
    if line.matches('*').count().is_multiple_of(2) {
        return (false, None);
    }

    let partial = line.rfind('*').map_or(line, |i| &line[i + 1..]);
    if partial.contains(' ') {
        return (true, Some(AsteriskClose::SpaceInPartial));
    }

    if partial.contains('\n') {
        return (true, Some(AsteriskClose::NewlineInPartial));
    }

    if py::len(partial) > ASTERISK_MAX_LEN {
        return (true, Some(AsteriskClose::TooLong));
    }

    (true, None)
}

/// `_get_unclosed(line)`: unbalanced `[ ]`, `< >` tags, `( )` and `*`
#[derive(Debug, Clone, Copy)]
pub struct Unclosed {
    pub bracket: bool,
    pub angle: bool,
    pub paren: bool,
    pub asterisk: bool,
}

impl Unclosed {
    pub fn of(line: &str) -> Self {
        let count = |c: char| line.matches(c).count();
        Self {
            bracket: count('[') != count(']'),
            angle: tag_opens(line).len() != tag_closes(line).len(),
            paren: count('(') != count(')'),
            asterisk: count('*') % 2 == 1,
        }
    }
}

/// The value of the last `<|t|>` in `s`, as `parse_timing(match.group(0))`
pub fn last_timing_in(s: &str) -> Option<f64> {
    TIMINGS_FORMAT
        .find_iter(s)
        .last()
        .and_then(|m| parse_timing(m.as_str()))
}

/// `_get_timing_range_cpu`: the `[start, end)` slice of `timing_token_ids` the next
/// timestamp may come from
pub fn timing_range(
    cur: &str,
    line: &str,
    max_timing: Option<f64>,
    min_timing: Option<f64>,
) -> (usize, usize) {
    let last_match = TIMINGS_FORMAT.find_iter(line).last();
    let (mut last_t, text_after) = match last_match {
        Some(m) => (parse_timing(m.as_str()), &line[m.end()..]),
        None => (None, ""),
    };

    if let Some(min) = min_timing
        && last_t.is_none_or(|t| t < min)
    {
        last_t = Some(min);
    }

    let max_idx = (MAX_TIMING_IDX + 1) as i64;
    let range = |start: f64, end: Option<f64>| -> (usize, usize) {
        let s = py::round(start / TIMING_STEP).max(0);
        let Some(mut end) = end else {
            return (s as usize, max_idx as usize);
        };
        if let Some(max) = max_timing
            && max < end
        {
            end = max;
        }

        let e = (py::round(end / TIMING_STEP) + 1).min(max_idx);
        // a negative end only ever meets `s < e`, which it fails
        (s as usize, e.max(0) as usize)
    };
    let cap_at_max = |cap: f64| match max_timing {
        Some(max) if max < cap => max,
        _ => cap,
    };

    let Some(last_t) = last_t else {
        if let Some(prev_t) = last_timing_in(cur) {
            // python's max(0, x) keeps the int 0 unless x is larger
            let start = if prev_t - 5.0 > 0.0 {
                prev_t - 5.0
            } else {
                0.0
            };
            return range(start, Some(cap_at_max(prev_t + 2.0)));
        }

        return range(0.0, Some(cap_at_max(1.0)));
    };

    let min_next = last_t + TIMING_STEP;
    let tail = if py::len(line) >= 6 {
        py::tail(line, 6)
    } else {
        ""
    };
    if tail.contains("pause]") || tail.contains("laugh]") {
        return range(min_next, Some(cap_at_max(last_t + 10.0)));
    }

    let leeway = count_words(text_after) as f64 * 1.0 + 1.0;
    range(min_next, Some(cap_at_max(last_t + leeway)))
}
