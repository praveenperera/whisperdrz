//! CPython semantics the post-processing depends on: `str` whitespace and word
//! classes, `float()`/`int()` on matched digit runs, `round()` and `f"{x:.2f}"`
//!
//! Byte offsets stand in for Python's code-point indices everywhere: every slice
//! boundary comes from a regex match or a char boundary, so the two index the same
//! text. Only `cur[:10]` in `finalize_transcription` counts code points, through
//! [`prefix_chars`]

use eyre::{Result, eyre};

use super::pyclass::{DIGIT, DIGIT_ZEROS, SPACE, WORD};

fn in_table(table: &[(u32, u32)], c: char) -> bool {
    let c = c as u32;
    table
        .binary_search_by(|&(lo, hi)| {
            if hi < c {
                std::cmp::Ordering::Less
            } else if lo > c {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// `str.isspace()`, which is also what `\s` matches; unlike `char::is_whitespace` it
/// includes the separators U+001C..U+001F
pub fn is_space(c: char) -> bool {
    in_table(SPACE, c)
}

/// What CPython's `\w` matches: `isalnum()` or `_`
pub fn is_word(c: char) -> bool {
    in_table(WORD, c)
}

/// What CPython's `\d` matches: Unicode category Nd at CPython's Unicode version
pub fn is_digit(c: char) -> bool {
    in_table(DIGIT, c)
}

/// The decimal value of a `\d` code point, as `int()` and `float()` read it
pub fn digit_value(c: char) -> Option<u32> {
    if !is_digit(c) {
        return None;
    }

    let c = c as u32;
    let i = DIGIT_ZEROS.partition_point(|&z| z <= c);
    let zero = DIGIT_ZEROS[i.checked_sub(1)?];
    (c - zero < 10).then_some(c - zero)
}

/// `str.strip()`
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// `str.lstrip()`
pub fn lstrip(s: &str) -> &str {
    s.trim_start_matches(is_space)
}

/// `str.rstrip()`
pub fn rstrip(s: &str) -> &str {
    s.trim_end_matches(is_space)
}

/// `s[:n]` with `n` counted in code points
pub fn prefix_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// Maps every Unicode decimal digit to its ASCII digit, as `float()` and `int()` do
fn ascii_digits(s: &str) -> String {
    s.chars()
        .map(|c| match digit_value(c) {
            Some(v) if !c.is_ascii_digit() => char::from(b'0' + v as u8),
            _ => c,
        })
        .collect()
}

/// `float(s)` for the strings the post-processing feeds it: decimal literals made of
/// `\d` digits and dots, possibly with an exponent, sign or surrounding whitespace.
/// Rust's parser is correctly rounded like CPython's. Fails where `float()` raises
/// `ValueError`
pub fn float(s: &str) -> Result<f64> {
    let t = ascii_digits(strip(s));
    // Rust accepts a few spellings CPython rejects and vice versa (`_` separators);
    // the reference only ever passes `\d+\.\d+` or `[\d.]+` captures here
    if t.contains('_') {
        return Err(eyre!("could not convert string to float: {s:?}"));
    }

    t.parse::<f64>().map_err(|_| eyre!("could not convert string to float: {s:?}"))
}

/// `int(s)` for a `\d+` capture; CPython ints are unbounded, so values past `i64`
/// are an error here
pub fn int(s: &str) -> Result<i64> {
    let mut v: i64 = 0;
    let mut any = false;
    for c in strip(s).chars() {
        let d = digit_value(c).ok_or_else(|| eyre!("invalid literal for int(): {s:?}"))?;
        v = v
            .checked_mul(10)
            .and_then(|v| v.checked_add(d as i64))
            .ok_or_else(|| eyre!("int({s:?}) is past i64"))?;
        any = true;
    }

    if !any {
        return Err(eyre!("invalid literal for int(): {s:?}"));
    }

    Ok(v)
}

/// `float(round(x))`: `round()` without digits returns an `int` (half to even), so a
/// result of zero is `+0.0` even for small negative inputs
pub fn round_int(x: f64) -> f64 {
    let r = x.round_ties_even();
    if r == 0.0 { 0.0 } else { r }
}

/// `f"{x:.2f}"`: the exact binary value correctly rounded, half to even, which is
/// also what Rust's precision formatting does
pub fn fmt_2f(x: f64) -> String {
    format!("{x:.2}")
}

/// `round(x, 2)`: CPython rounds the exact binary value to two decimals (half to
/// even on exact ties) and reads the decimal back, the same as formatting and
/// parsing
pub fn round2(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }

    fmt_2f(x).parse().unwrap_or(x)
}

/// `min(a, b)` for floats: the first argument unless the second is smaller
pub fn min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// `max(a, b)` for floats: the first argument unless the second is larger
pub fn max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digits_and_space_follow_cpython() {
        assert_eq!(digit_value('٣'), Some(3));
        assert_eq!(digit_value('9'), Some(9));
        assert_eq!(digit_value('²'), None);
        assert!(is_space('\u{1c}'));
        assert!(!'\u{1c}'.is_whitespace());
        assert!(is_word('²'));
        assert!(!is_word('\u{301}'));
        assert_eq!(float("١.٥").unwrap(), 1.5);
        assert_eq!(int("٤2").unwrap(), 42);
    }
}
