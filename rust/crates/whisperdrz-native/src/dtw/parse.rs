//! Window-text parsing into speaker lines, words and timed tag anchors
//! (`parse_predicted_lines`, `_extract_line_words_and_tag_anchors`)

use std::sync::LazyLock;

use eyre::{Result, WrapErr};
use regex::Regex;

// Python's `\s` and `str.isspace()` count the ASCII separators U+001C..U+001F as
// whitespace; Unicode `White_Space` (Rust's `\s` and `char::is_whitespace`) does not,
// so every whitespace class below adds them
static TOKEN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<\|(\d+\.\d+)\|>|[^\s\x1c-\x1f<]+").unwrap());
// `<(?!\|)[^>]*>` without the lookahead: `<`, then nothing or a first char that is
// neither `|` nor `>`, then anything up to the first `>`
static STRIP_META: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<(?:[^|>][^>]*)?>").unwrap());
pub(super) static STRIP_CHOP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[CHOP\]").unwrap());
static SPK_LINE_FIND: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[(\d+|c)\][\s\x1c-\x1f]*").unwrap());

/// `str.isspace()` for one char
pub fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

/// A speaker label: Python keys lines by `int(label)`, or by the string `"c"` for crosstalk
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Speaker {
    /// The label's integer value in decimal, so `[07]` and `[7]` are the same speaker
    Num(String),
    Crosstalk,
}

impl Speaker {
    fn from_label(label: &str) -> Self {
        if label == "c" {
            return Self::Crosstalk;
        }

        // non-ASCII decimal digits stay as written (Python would take their value)
        if !label.bytes().all(|b| b.is_ascii_digit()) {
            return Self::Num(label.to_owned());
        }

        let trimmed = label.trim_start_matches('0');
        Self::Num(if trimmed.is_empty() {
            "0".to_owned()
        } else {
            trimmed.to_owned()
        })
    }
}

/// A word with the decoder timestamps glued to it, if any
#[derive(Debug, Clone, PartialEq)]
pub struct PredWord {
    pub text: String,
    pub start: Option<f64>,
    pub end: Option<f64>,
}

/// One speaker turn: its label and the indices of its words in [`ParsedText::words`]
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerLine {
    pub speaker: Speaker,
    pub words: Vec<usize>,
}

/// A timed `<|s|>[tag]<|e|>` and the words around it, at least one of them set
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TagAnchor {
    pub start: f64,
    pub end: f64,
    pub prev: Option<usize>,
    pub next: Option<usize>,
}

/// A window text split for DTW
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ParsedText {
    pub lines: Vec<SpeakerLine>,
    /// Every word in text order (`pred_words_flat`)
    pub words: Vec<PredWord>,
    /// Tag anchors with indices into `words`
    pub tag_anchors: Vec<TagAnchor>,
}

/// Splits a window text into speaker lines, a flat word list and timed tag anchors,
/// as `parse_predicted_lines`
///
/// Errors only on a timestamp whose digits aren't ASCII, which `float()` would accept
pub fn parse_predicted_lines(text: &str) -> Result<ParsedText> {
    let text = STRIP_CHOP.replace_all(text, " ");
    let matches: Vec<_> = SPK_LINE_FIND.captures_iter(&text).collect();
    let mut out = ParsedText::default();

    let first_start = matches.first().map(|m| m.get(0).unwrap().start());
    if first_start != Some(0) {
        let end = first_start.unwrap_or(text.len());
        out.consume(&text[..end], Speaker::Num("0".to_owned()))?;
    }

    for (i, m) in matches.iter().enumerate() {
        let end = matches
            .get(i + 1)
            .map_or(text.len(), |n| n.get(0).unwrap().start());
        let speaker = Speaker::from_label(m.get(1).unwrap().as_str());
        out.consume(&text[m.get(0).unwrap().end()..end], speaker)?;
    }

    Ok(out)
}

impl ParsedText {
    fn consume(&mut self, sub: &str, speaker: Speaker) -> Result<()> {
        let (words, tags) = extract_line_words_and_tag_anchors(sub)?;
        if words.is_empty() && tags.is_empty() {
            return Ok(());
        }

        let first = self.words.len();
        let n = words.len();
        self.words.extend(words);
        if n > 0 {
            self.lines.push(SpeakerLine {
                speaker,
                words: (first..first + n).collect(),
            });
        }

        let global = |li: Option<usize>| li.filter(|&li| li < n).map(|li| first + li);
        for tag in tags {
            let (prev, next) = (global(tag.prev), global(tag.next));
            if prev.is_none() && next.is_none() {
                continue;
            }

            self.tag_anchors.push(TagAnchor { prev, next, ..tag });
        }

        Ok(())
    }
}

/// Words and timed tag anchors (with line-local word indices) of one line's text
fn extract_line_words_and_tag_anchors(line_text: &str) -> Result<(Vec<PredWord>, Vec<TagAnchor>)> {
    struct Item<'a> {
        tok: &'a str,
        start: usize,
        end: usize,
        time: Option<f64>,
    }

    let chopped = STRIP_CHOP.replace_all(line_text, " ");
    let text = STRIP_META.replace_all(&chopped, " ");
    let items = TOKEN_RE
        .captures_iter(&text)
        .map(|c| {
            let m = c.get(0).unwrap();
            let time = c.get(1).map(|t| parse_seconds(t.as_str())).transpose()?;
            Ok(Item {
                tok: m.as_str(),
                start: m.start(),
                end: m.end(),
                time,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut words = Vec::new();
    let mut tags = Vec::new();
    for (i, item) in items.iter().enumerate() {
        if item.time.is_some() {
            continue;
        }

        // timestamps directly against the token, with no whitespace between
        let left = i
            .checked_sub(1)
            .map(|k| &items[k])
            .filter(|p| p.end == item.start)
            .and_then(|p| p.time);
        let right = items
            .get(i + 1)
            .filter(|n| n.start == item.end)
            .and_then(|n| n.time);

        let tok = item.tok;
        if tok.starts_with('[') && tok.ends_with(']') && tok.chars().count() > 2 {
            // a timed `[tag]` anchors its neighbours; untimed tags are skipped
            if let (Some(start), Some(end)) = (left, right) {
                let prev = words.len().checked_sub(1);
                tags.push(TagAnchor {
                    start,
                    end,
                    prev,
                    next: Some(words.len()),
                });
            }

            continue;
        }

        words.push(PredWord {
            text: tok.to_owned(),
            start: left,
            end: right,
        });
    }

    let n = words.len();
    let tags = tags
        .into_iter()
        .map(|t| TagAnchor {
            next: t.next.filter(|&k| k < n),
            ..t
        })
        .filter(|t| t.prev.is_some() || t.next.is_some())
        .collect();
    Ok((words, tags))
}

fn parse_seconds(s: &str) -> Result<f64> {
    // `\d` also matches non-ASCII decimal digits, which Rust's parser rejects
    s.parse()
        .wrap_err_with(|| format!("timestamp {s:?} is not an ASCII decimal"))
}
