//! Rewriting a window text with DTW timings (`rebuild_with_timings`)

use std::sync::LazyLock;

use eyre::{Result, ensure};
use regex::Regex;

use super::parse::{PredWord, is_py_space};

// bracket and speaker tags before bare words so they aren't absorbed; whitespace
// classes add U+001C..U+001F, which Python's `\s` includes. Text no alternative
// matches (a lone `[` or `<`) is dropped, as `finditer` skips it
static PARSE_TOKEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"\[/?[a-zA-Z]+\]|\[(?:\d+|c)\]|<\|\d+\.\d+\|>|<[^>]+>|[^\s\x1c-\x1f<\[]+|[\s\x1c-\x1f]+",
    )
    .unwrap()
});

/// Wraps each untimed word that has a DTW timing in `<|s|>word<|e|>`, as
/// `rebuild_with_timings`
///
/// Walks the original text so tags, metadata, whitespace and decoder timings stay
/// verbatim; the n-th bare word takes `words[n]` and `timings[n]`. Timings snap to the
/// 0.02 s grid. Errors when the two lists differ in length or a timing isn't finite
pub fn rebuild_with_timings(
    pred_text: &str,
    words: &[PredWord],
    timings: &[Option<(f64, f64)>],
) -> Result<String> {
    ensure!(
        words.len() == timings.len(),
        "{} words but {} timings",
        words.len(),
        timings.len()
    );

    let mut out = String::with_capacity(pred_text.len() + 16 * words.len());
    let mut word_idx = 0;
    for m in PARSE_TOKEN.find_iter(pred_text) {
        let s = m.as_str();
        // only the bare-word alternative can start with anything else
        let first = s.chars().next().unwrap();
        if first == '<' || first == '[' || is_py_space(first) || word_idx >= words.len() {
            out.push_str(s);
            continue;
        }

        let w = &words[word_idx];
        word_idx += 1;
        let Some((ts, te)) = timings[word_idx - 1].filter(|_| w.start.is_none() && w.end.is_none())
        else {
            out.push_str(s);
            continue;
        };

        // `round()` raises on non-finite values
        ensure!(
            ts.is_finite() && te.is_finite(),
            "non-finite timing ({ts}, {te}) for {s:?}"
        );
        let ts = py_max(0.0, snap(ts));
        let te = py_max(ts, snap(te));
        out.push_str(&format!("<|{ts:.2}|>{s}<|{te:.2}|>"));
    }

    Ok(out)
}

/// `round(t * 50) / 50`: half-to-even to an integer, then true division
fn snap(t: f64) -> f64 {
    (t * 50.0).round_ties_even() / 50.0
}

/// Python's `max(a, b)`: `b` only when strictly greater
fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}
