//! `clean/timings.py`: parsing, quantizing, shifting and repairing `<|t|>` markers

use eyre::Result;

use super::pattern::{Lazy, PyRe};
use super::py;

/// `TIMINGS_FORMAT`: `<|1.23|>`, capturing `1.23`
pub static TIMINGS_FORMAT: Lazy = Lazy::new(|| PyRe::plain(r"<\|(\d+\.\d+)\|>"));
/// `SCORED_TIMING_PATTERN`: `<a.5|1.23|>`, capturing the score and the time
static SCORED_TIMING: Lazy = Lazy::new(|| PyRe::plain(r"<a\.(\d+)\|(\d+\.\d+)\|>"));
static TIMED_WORD: Lazy = Lazy::new(|| PyRe::plain(r"<\|\d+\.\d+\|>([^\s<\[]+)<\|\d+\.\d+\|>"));
static DECODER_ALIGN: Lazy = Lazy::new(|| {
    PyRe::plain(
        r"(<\|\d+\.\d+\|>)(<a\.\d+\|\d+\.\d+\|>)(.*?)(<a\.\d+\|\d+\.\d+\|>)(<\|\d+\.\d+\|>)",
    )
});
static ALIGN_SCORE: Lazy = Lazy::new(|| PyRe::plain(r"<a\.(\d+)\|"));
static ALIGN_TIME: Lazy = Lazy::new(|| PyRe::plain(r"<a\.\d+\|(\d+\.\d+)\|>"));
static SHORT_NONSPEECH: Lazy =
    Lazy::new(|| PyRe::plain(r"<\|([\d.]+)\|>\s*(\[[a-z]+\])\s*<\|([\d.]+)\|>"));
static SHORT_PAUSE: Lazy =
    Lazy::new(|| PyRe::plain(r"<\|([\d.]+)\|>\s*\[pause\]\s*<\|([\d.]+)\|>"));

/// `parse_timing`: `float` of a marker with `<|` and `|>` removed
pub fn parse_timing(t: &str) -> Result<f64> {
    py::float(&t.replace("<|", "").replace("|>", ""))
}

/// `cut_after_last_timed_word`: drops everything after the last `<|a|>word<|b|>`
pub fn cut_after_last_timed_word(input: &str) -> Result<String> {
    Ok(match TIMED_WORD.captures_all(input)?.last() {
        Some(m) => input[..m.end()].to_owned(),
        None => input.to_owned(),
    })
}

/// Whether a run of `\d` digits reads as an integer at or above `threshold`
fn digits_at_least(digits: &str, threshold: i64) -> Result<bool> {
    // `int()` is unbounded in Python, so a run too long for i64 is past any threshold
    Ok(match py::int(digits) {
        Ok(v) => v >= threshold,
        Err(_) if digits.chars().all(py::is_digit) => true,
        Err(e) => return Err(e),
    })
}

/// `reconcile_decoder_and_alignment_timings`: in `<|d|><a.s|a|>word<a.s|a|><|d|>`,
/// keeps the alignment times when the score reaches `confidence_threshold` and the
/// decoder times otherwise; then rewrites every leftover `<a.s|t|>` as `<|t|>`
pub fn reconcile_decoder_and_alignment_timings(
    text: &str,
    confidence_threshold: i64,
) -> Result<String> {
    let result = DECODER_ALIGN.try_sub_with(text, |m| {
        let (decoder_start, align_start, word, align_end, decoder_end) =
            (m.group(1), m.group(2), m.group(3), m.group(4), m.group(5));
        let Some(score) = ALIGN_SCORE.search(align_start)? else {
            return Ok(m.as_str().to_owned());
        };

        if digits_at_least(score.group(1), confidence_threshold)? {
            let start = ALIGN_TIME
                .search(align_start)?
                .map(|c| c.group(1))
                .unwrap_or_default();
            let end = ALIGN_TIME
                .search(align_end)?
                .map(|c| c.group(1))
                .unwrap_or_default();
            return Ok(format!("<|{start}|>{word}<|{end}|>"));
        }

        Ok(format!("{decoder_start}{word}{decoder_end}"))
    })?;

    SCORED_TIMING.try_sub_with(&result, |m| {
        Ok(format!("<|{}|>", py::fmt_2f(py::float(m.group(2))?)))
    })
}

/// `round_to`: `round(x / step) * step`, where `round` returns an int
pub fn round_to(x: f64, step: f64) -> f64 {
    py::round_int(x / step) * step
}

/// `quantize_timestamp` for one captured time string
pub fn quantize_timestamp(time: &str, step: f64, offset: f64) -> Result<String> {
    let time = py::float(time)?;
    let mut quantized = round_to(time, step);
    let offset = round_to(offset, step);
    quantized += offset;
    if offset == 0.0 {
        quantized = py::min(quantized, 30.0);
    }

    Ok(format!("<|{}|>", py::fmt_2f(quantized)))
}

/// `quantize_timestamps`: snaps every marker to `step` and adds `offset`; without
/// an offset, times are capped at 30 s
pub fn quantize_timestamps(text: &str, offset: f64, step: f64) -> Result<String> {
    TIMINGS_FORMAT.try_sub_with(text, |m| quantize_timestamp(m.group(1), step, offset))
}

/// `offset_timestamps`: shifts every marker by `offset`, floored at 0 and snapped to
/// 0.02 s
pub fn offset_timestamps(text: &str, offset: f64) -> Result<String> {
    TIMINGS_FORMAT.try_sub_with(text, |m| {
        let time = py::float(m.group(1))?;
        let new_time = round_to(py::max(0.0, time + offset), 0.02);
        Ok(format!("<|{}|>", py::fmt_2f(new_time)))
    })
}

/// `remove_invalid_timing_pairs`: drops both markers of each pair (consecutive
/// markers on one line with at most two spaces between) that lasts `min_duration`
/// or less; unpaired markers stay
pub fn remove_invalid_timing_pairs(text: &str, min_duration: f64) -> Result<String> {
    let matches = TIMINGS_FORMAT.captures_all(text)?;
    if matches.len() < 2 {
        return Ok(text.to_owned());
    }

    let mut used = vec![false; matches.len()];
    let mut remove = Vec::new();
    for i in 0..matches.len() - 1 {
        if used[i] {
            continue;
        }

        let (t1, t2) = (&matches[i], &matches[i + 1]);
        let between = &text[t1.end()..t2.start()];
        if between.matches(' ').count() > 2 || between.contains('\n') {
            continue;
        }

        used[i] = true;
        used[i + 1] = true;
        let duration = py::float(t2.group(1))? - py::float(t1.group(1))?;
        if duration <= min_duration {
            remove.push((t1.start(), t1.end(), t2.start(), t2.end()));
        }
    }

    // pairs never overlap, so cutting from the back keeps earlier offsets valid
    let mut out = text.to_owned();
    for &(s1, e1, s2, e2) in remove.iter().rev() {
        out.replace_range(s2..e2, "");
        out.replace_range(s1..e1, "");
    }

    Ok(out)
}

/// `fix_nonmonotonic_timestamps`: within each line, a marker earlier than the one
/// before it becomes that one plus 0.02 s
pub fn fix_nonmonotonic_timestamps(text: &str) -> Result<String> {
    let mut lines = Vec::new();
    for line in text.split('\n') {
        let matches = TIMINGS_FORMAT.captures_all(line)?;
        if matches.len() < 2 {
            lines.push(line.to_owned());
            continue;
        }

        let mut last_t = py::float(matches[0].group(1))?;
        let mut replacements = Vec::new();
        for m in &matches[1..] {
            let t = py::float(m.group(1))?;
            if t < last_t {
                let new_t = last_t + 0.02;
                replacements.push((m.start(), m.end(), format!("<|{}|>", py::fmt_2f(new_t))));
                last_t = new_t;
            } else {
                last_t = t;
            }
        }

        let mut line = line.to_owned();
        for (start, end, repl) in replacements.iter().rev() {
            line.replace_range(*start..*end, repl);
        }

        lines.push(line);
    }

    Ok(lines.join("\n"))
}

/// `remove_short_nonspeech`: drops `<|a|>[tag]<|b|>` when `b - a` is at most
/// `min_duration_s` (plus a 1 ms epsilon)
pub fn remove_short_nonspeech(text: &str, min_duration_s: f64) -> Result<String> {
    SHORT_NONSPEECH.try_sub_with(text, |m| {
        let t1 = py::float(m.group(1))?;
        let t2 = py::float(m.group(3))?;
        Ok(if t2 - t1 <= min_duration_s + 0.001 {
            String::new()
        } else {
            m.as_str().to_owned()
        })
    })
}

/// `remove_short_pauses`: replaces `<|a|>[pause]<|b|>` shorter than `min_pause_ms`
/// with `<|b|>`
pub fn remove_short_pauses(text: &str, min_pause_ms: i64) -> Result<String> {
    let min_pause_s = min_pause_ms as f64 / 1000.0;
    SHORT_PAUSE.try_sub_with(text, |m| {
        let t1 = py::float(m.group(1))?;
        let t2 = py::float(m.group(2))?;
        Ok(if t2 - t1 < min_pause_s {
            format!("<|{}|>", py::fmt_2f(t2))
        } else {
            m.as_str().to_owned()
        })
    })
}
