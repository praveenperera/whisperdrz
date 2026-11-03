//! Window stitching and the `segments` list: `_strip_to_last_closing` and
//! `parse_to_segments` from `transcriber.py`

use std::fmt;

use eyre::Result;
use serde::ser::{Serialize, SerializeMap, Serializer};

use super::pattern::{Lazy, PyRe};
use super::py;
use super::speakers::SPEAKER_LINE_START;
use super::timings::TIMINGS_FORMAT;

static WHITESPACE_RUN: Lazy = Lazy::new(|| PyRe::plain(r"\s+"));

/// `_strip_to_last_closing`: cuts the text after the last closing marker (markers
/// alternate open/close on a line, so an even count ends on a closing one). Lines
/// at the end holding a single orphaned opening marker are dropped; with no
/// markers at all the text comes back unchanged
pub fn strip_to_last_closing(cur: &str) -> Result<String> {
    let lines: Vec<&str> = cur.split('\n').collect();
    for i in (0..lines.len()).rev() {
        let ts = TIMINGS_FORMAT.captures_all(lines[i])?;
        let cut = match ts.len() {
            0 | 1 => continue,
            n if n % 2 == 0 => ts[n - 1].end(),
            n => ts[n - 2].end(),
        };

        let mut out = lines[..i].join("\n");
        if i > 0 {
            out.push('\n');
        }

        out.push_str(&lines[i][..cut]);
        return Ok(out);
    }

    Ok(cur.to_owned())
}

/// A segment's speaker: the `[N]` id, or `"c"` for a crowd line
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Speaker {
    Id(i64),
    Crowd,
}

impl Speaker {
    /// Parses the `\d+|c` capture of a speaker tag
    pub fn parse(s: &str) -> Result<Self> {
        if s == "c" { Ok(Self::Crowd) } else { py::int(s).map(Self::Id) }
    }
}

impl fmt::Display for Speaker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Id(n) => write!(f, "{n}"),
            Self::Crowd => f.write_str("c"),
        }
    }
}

impl Serialize for Speaker {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::Id(n) => s.serialize_i64(*n),
            Self::Crowd => s.serialize_str("c"),
        }
    }
}

/// One line of the final transcript, as `parse_to_segments` builds it
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub speaker: Speaker,
    /// First marker, `round(t, 2)`
    pub start: f64,
    /// Last marker, `round(t, 2)`
    pub end: f64,
    /// The words with markers removed and whitespace runs collapsed
    pub text: String,
}

impl Serialize for Segment {
    /// The Python dict's key order: speaker, start, end, text
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(4))?;
        map.serialize_entry("speaker", &self.speaker)?;
        map.serialize_entry("start", &self.start)?;
        map.serialize_entry("end", &self.end)?;
        map.serialize_entry("text", &self.text)?;
        map.end()
    }
}

/// `parse_to_segments`: one segment per line with a speaker tag (default 0) and at
/// least two markers ending after they start; `[CHOP]` lines are skipped
pub fn parse_to_segments(text: &str) -> Result<Vec<Segment>> {
    let mut segments = Vec::new();
    for line in text.split('\n') {
        let mut line = py::strip(line);
        if line.is_empty() || line.starts_with("[CHOP]") {
            continue;
        }

        let mut speaker = Speaker::Id(0);
        if let Some(m) = SPEAKER_LINE_START.search(line)? {
            speaker = Speaker::parse(m.group(1))?;
            line = &line[m.end()..];
        }

        let timings = TIMINGS_FORMAT.captures_all(line)?;
        if timings.len() < 2 {
            continue;
        }

        let start = py::float(timings[0].group(1))?;
        let end = py::float(timings[timings.len() - 1].group(1))?;
        if end <= start {
            continue;
        }

        let words = TIMINGS_FORMAT.sub(line, "")?;
        let text = WHITESPACE_RUN.sub(py::strip(&words), " ")?;
        segments.push(Segment { speaker, start: py::round2(start), end: py::round2(end), text });
    }

    Ok(segments)
}
