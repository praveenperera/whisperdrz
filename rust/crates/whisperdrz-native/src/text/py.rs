//! Python `str` semantics the decoding rules depend on
//!
//! CPython 3.12 (Unicode 15.0) is the reference. Character classes come from
//! `unicode-general-category`, which may carry a newer Unicode version; the vocab tests
//! check every character that occurs in a token against Python, which is the only text
//! the decoder ever sees

use unicode_general_category::{GeneralCategory, get_general_category};

/// Characters for which `str.isspace()` is true (note `\x1c`..`\x1f`, which Rust's
/// `char::is_whitespace` leaves out)
pub fn is_space(c: char) -> bool {
    matches!(
        c,
        '\t'..='\r'
            | '\x1c'..='\x1f'
            | ' '
            | '\u{85}'
            | '\u{a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}'
    )
}

fn category(c: char) -> GeneralCategory {
    get_general_category(c)
}

/// `str.isalpha()`: general category L*
pub fn is_alpha(c: char) -> bool {
    use GeneralCategory::{
        LowercaseLetter, ModifierLetter, OtherLetter, TitlecaseLetter, UppercaseLetter,
    };

    matches!(
        category(c),
        UppercaseLetter | LowercaseLetter | TitlecaseLetter | ModifierLetter | OtherLetter
    )
}

/// `str.isdecimal()`: general category Nd
pub fn is_decimal(c: char) -> bool {
    category(c) == GeneralCategory::DecimalNumber
}

/// Code points with `Numeric_Type=Digit` (not decimal), from CPython 3.12's
/// `unicodedata`; `str.isdigit()` is true for these and for every Nd character
const DIGIT_NOT_DECIMAL: &[(u32, u32)] = &[
    (0x00B2, 0x00B3),
    (0x00B9, 0x00B9),
    (0x1369, 0x1371),
    (0x19DA, 0x19DA),
    (0x2070, 0x2070),
    (0x2074, 0x2079),
    (0x2080, 0x2089),
    (0x2460, 0x2468),
    (0x2474, 0x247C),
    (0x2488, 0x2490),
    (0x24EA, 0x24EA),
    (0x24F5, 0x24FD),
    (0x24FF, 0x24FF),
    (0x2776, 0x277E),
    (0x2780, 0x2788),
    (0x278A, 0x2792),
    (0x10A40, 0x10A43),
    (0x10E60, 0x10E68),
    (0x11052, 0x1105A),
    (0x1F100, 0x1F10A),
];

/// `str.isdigit()`
pub fn is_digit(c: char) -> bool {
    is_decimal(c)
        || DIGIT_NOT_DECIMAL
            .iter()
            .any(|&(lo, hi)| (lo..=hi).contains(&(c as u32)))
}

/// `str.isalnum()`, which in Unicode 15.0 is exactly general category L* or N*
pub fn is_alnum(c: char) -> bool {
    use GeneralCategory::{DecimalNumber, LetterNumber, OtherNumber};

    is_alpha(c) || matches!(category(c), DecimalNumber | LetterNumber | OtherNumber)
}

/// `unicodedata.category(c).startswith("M")`
pub fn is_mark(c: char) -> bool {
    use GeneralCategory::{EnclosingMark, NonspacingMark, SpacingMark};

    matches!(category(c), NonspacingMark | SpacingMark | EnclosingMark)
}

/// `str.isprintable()` for one character, as `repr` uses it
pub fn is_printable(c: char) -> bool {
    use GeneralCategory::{
        Control, Format, LineSeparator, ParagraphSeparator, PrivateUse, SpaceSeparator, Surrogate,
        Unassigned,
    };

    if c == ' ' {
        return true;
    }

    !matches!(
        category(c),
        Control
            | Format
            | Surrogate
            | PrivateUse
            | Unassigned
            | LineSeparator
            | ParagraphSeparator
            | SpaceSeparator
    )
}

/// `str.strip()` with no arguments
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// `str.lstrip()` with no arguments
pub fn lstrip(s: &str) -> &str {
    s.trim_start_matches(is_space)
}

/// `str.rstrip()` with no arguments
pub fn rstrip(s: &str) -> &str {
    s.trim_end_matches(is_space)
}

/// `str.split()` with no arguments: runs of whitespace separate words, no empty words
pub fn split_whitespace(s: &str) -> impl Iterator<Item = &str> {
    s.split(is_space).filter(|w| !w.is_empty())
}

/// `len(s)`: code points, not bytes
pub fn len(s: &str) -> usize {
    s.chars().count()
}

/// `s[-n:]`: the last `n` code points
pub fn tail(s: &str, n: usize) -> &str {
    match s.char_indices().rev().nth(n.saturating_sub(1)) {
        Some((i, _)) if n > 0 => &s[i..],
        _ if n == 0 => "",
        _ => s,
    }
}

/// `s[-1]` (or `None` for an empty string)
pub fn last_char(s: &str) -> Option<char> {
    s.chars().next_back()
}

/// `float(s)` for the strings the decoder parses: optional sign, decimal digits
/// (any Unicode Nd digit, as Python allows), one optional point, optional exponent,
/// `inf`/`nan`, surrounding whitespace and single underscores between digits.
/// Returns `None` where Python raises `ValueError`
pub fn float(s: &str) -> Option<f64> {
    let s = strip(s);
    let mut ascii = String::with_capacity(s.len());
    let mut prev_digit = false;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '_' {
            // python allows `_` only between two digits
            let next_digit = chars.peek().is_some_and(|&n| decimal_value(n).is_some());
            if !(prev_digit && next_digit) {
                return None;
            }

            continue;
        }

        match decimal_value(c) {
            Some(d) => {
                ascii.push(char::from(b'0' + d));
                prev_digit = true;
            }
            None => {
                ascii.push(c);
                prev_digit = false;
            }
        }
    }

    // rust's grammar is python's once underscores and non-ASCII digits are gone:
    // optional sign, digits with one optional point, optional exponent, inf/infinity/nan
    if ascii.is_empty() || ascii.contains(char::is_whitespace) {
        return None;
    }

    ascii.parse().ok()
}

/// The value of a decimal digit (`str.isdecimal()`), ASCII or any other Nd digit
fn decimal_value(c: char) -> Option<u8> {
    if c.is_ascii_digit() {
        return Some(c as u8 - b'0');
    }

    if !is_decimal(c) {
        return None;
    }

    // unicode encodes every decimal digit set as a contiguous run from 0 to 9
    let mut value = 0u8;
    let mut cp = c as u32;
    while value < 9 {
        match cp.checked_sub(1).and_then(char::from_u32) {
            Some(prev) if is_decimal(prev) => {
                value += 1;
                cp -= 1;
            }
            _ => break,
        }
    }

    Some(value)
}

/// `repr(s)` for a `str`
pub fn repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ if c == quote => {
                out.push('\\');
                out.push(c);
            }
            _ if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            _ if c.is_ascii() || is_printable(c) => out.push(c),
            _ if (c as u32) <= 0xff => out.push_str(&format!("\\x{:02x}", c as u32)),
            _ if (c as u32) <= 0xffff => out.push_str(&format!("\\u{:04x}", c as u32)),
            _ => out.push_str(&format!("\\U{:08x}", c as u32)),
        }
    }

    out.push(quote);
    out
}

/// `round(x)` for a float: half to even on the exact binary value
pub fn round(x: f64) -> i64 {
    x.round_ties_even() as i64
}

#[cfg(test)]
mod tests {
    use super::{float, repr, tail};

    #[test]
    fn float_matches_python_grammar() {
        // expected values from CPython 3.12 `float()`
        assert_eq!(float(" 0.12 "), Some(0.12));
        assert_eq!(float("1_0.5"), Some(10.5));
        assert_eq!(float("\u{0661}.\u{0662}"), Some(1.2));
        assert_eq!(float("-inf"), Some(f64::NEG_INFINITY));
        assert_eq!(float("1e3"), Some(1000.0));
        assert_eq!(float("."), None);
        assert_eq!(float("1__0"), None);
        assert_eq!(float("_1"), None);
        assert_eq!(float("en"), None);
    }

    #[test]
    fn repr_matches_python() {
        // expected values from CPython 3.12 `repr()`
        assert_eq!(repr("it's"), "\"it's\"");
        assert_eq!(repr("a'b\"c"), "'a\\'b\"c'");
        assert_eq!(repr(" so\n"), "' so\\n'");
        assert_eq!(
            repr("\u{fffd}\u{200b}\u{a0}é\u{7f}"),
            "'\u{fffd}\\u200b\\xa0é\\x7f'"
        );
    }

    #[test]
    fn tail_counts_code_points() {
        assert_eq!(tail("aé[pause]", 6), "pause]");
        assert_eq!(tail("ab", 6), "ab");
        assert_eq!(tail("ab", 0), "");
    }
}
