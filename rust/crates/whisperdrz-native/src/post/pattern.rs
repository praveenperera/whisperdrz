//! Python `re` patterns on the Rust regex crates
//!
//! Patterns are written in Python syntax and translated once: `\s` and `\d` become
//! explicit classes built from CPython's tables (the crates' own classes follow a
//! newer Unicode version and leave out U+001C..U+001F from `\s`). Patterns with
//! lookaround or backreferences go to `fancy-regex`, the rest to `regex`; both pick
//! the same leftmost-first match a backtracking engine like `sre` does. No pattern
//! here can match the empty string, so the crates' empty-match rules never come
//! into play

use std::fmt::Write;

use eyre::{Result, WrapErr};

use super::pyclass::{DIGIT, SPACE};

/// The span of one capture group, as byte offsets
pub type Span = Option<(usize, usize)>;

/// One match with its groups; group 0 is the whole match
#[derive(Debug, Clone)]
pub struct Caps<'t> {
    text: &'t str,
    spans: Vec<Span>,
}

impl<'t> Caps<'t> {
    /// `m.group(i)`, or `None` when the group didn't take part
    pub fn get(&self, i: usize) -> Option<&'t str> {
        self.spans
            .get(i)
            .copied()
            .flatten()
            .map(|(a, b)| &self.text[a..b])
    }

    /// `m.group(i)` for a group the pattern always sets
    pub fn group(&self, i: usize) -> &'t str {
        self.get(i).unwrap_or_default()
    }

    pub fn start(&self) -> usize {
        self.spans[0].map_or(0, |s| s.0)
    }

    pub fn end(&self) -> usize {
        self.spans[0].map_or(0, |s| s.1)
    }

    pub fn as_str(&self) -> &'t str {
        &self.text[self.start()..self.end()]
    }
}

enum Engine {
    Plain(regex::Regex),
    Fancy(fancy_regex::Regex),
}

/// A compiled Python pattern
pub struct PyRe(Engine);

fn class_body(table: &[(u32, u32)]) -> String {
    let mut out = String::new();
    for &(a, b) in table {
        if a == b {
            let _ = write!(out, "\\x{{{a:X}}}");
        } else {
            let _ = write!(out, "\\x{{{a:X}}}-\\x{{{b:X}}}");
        }
    }

    out
}

/// Rewrites `\s` and `\d` into CPython's classes, inside and outside `[...]`
fn translate(src: &str) -> String {
    let space = class_body(SPACE);
    let digit = class_body(DIGIT);
    let mut out = String::with_capacity(src.len());
    let mut in_class = false;
    let mut chars = src.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                let Some(next) = chars.next() else {
                    out.push('\\');
                    break;
                };
                let body = match next {
                    's' => Some(&space),
                    'd' => Some(&digit),
                    _ => None,
                };
                match body {
                    Some(body) if in_class => out.push_str(body),
                    Some(body) => {
                        out.push('[');
                        out.push_str(body);
                        out.push(']');
                    }
                    None => {
                        out.push('\\');
                        out.push(next);
                    }
                }
            }
            '[' if !in_class => {
                in_class = true;
                out.push('[');
                // a leading `^` or `]` belongs to the class
                if chars.peek() == Some(&'^') {
                    out.push(chars.next().unwrap_or('^'));
                }
            }
            ']' if in_class => {
                in_class = false;
                out.push(']');
            }
            _ => out.push(c),
        }
    }

    out
}

impl PyRe {
    /// A pattern without lookaround or backreferences
    pub fn plain(src: &str) -> Self {
        let re = regex::Regex::new(&translate(src))
            .unwrap_or_else(|e| panic!("bad pattern {src:?}: {e}"));
        Self(Engine::Plain(re))
    }

    /// A pattern with lookaround or backreferences
    pub fn fancy(src: &str) -> Self {
        let re = fancy_regex::RegexBuilder::new(&translate(src))
            .backtrack_limit(100_000_000)
            .build()
            .unwrap_or_else(|e| panic!("bad pattern {src:?}: {e}"));
        Self(Engine::Fancy(re))
    }

    /// `finditer`: every non-overlapping match, left to right
    pub fn captures_all<'t>(&self, text: &'t str) -> Result<Vec<Caps<'t>>> {
        let mut out = Vec::new();
        match &self.0 {
            Engine::Plain(re) => {
                for c in re.captures_iter(text) {
                    let spans = (0..c.len())
                        .map(|i| c.get(i).map(|m| (m.start(), m.end())))
                        .collect();
                    out.push(Caps { text, spans });
                }
            }
            Engine::Fancy(re) => {
                for c in re.captures_iter(text) {
                    let c = c.wrap_err("regex match")?;
                    let spans = (0..c.len())
                        .map(|i| c.get(i).map(|m| (m.start(), m.end())))
                        .collect();
                    out.push(Caps { text, spans });
                }
            }
        }

        Ok(out)
    }

    /// `search`: the first match
    pub fn search<'t>(&self, text: &'t str) -> Result<Option<Caps<'t>>> {
        self.search_from(text, 0)
    }

    /// The first match starting at or after byte `pos`, with the text before `pos`
    /// still visible to anchors and lookbehind
    pub fn search_from<'t>(&self, text: &'t str, pos: usize) -> Result<Option<Caps<'t>>> {
        Ok(match &self.0 {
            Engine::Plain(re) => re.captures_at(text, pos).map(|c| Caps {
                text,
                spans: (0..c.len())
                    .map(|i| c.get(i).map(|m| (m.start(), m.end())))
                    .collect(),
            }),
            Engine::Fancy(re) => re
                .captures_from_pos(text, pos)
                .wrap_err("regex match")?
                .map(|c| Caps {
                    text,
                    spans: (0..c.len())
                        .map(|i| c.get(i).map(|m| (m.start(), m.end())))
                        .collect(),
                }),
        })
    }

    /// `re.sub` with a function replacement
    pub fn sub_with(&self, text: &str, mut repl: impl FnMut(&Caps) -> String) -> Result<String> {
        let mut out = String::with_capacity(text.len());
        let mut last = 0;
        for c in self.captures_all(text)? {
            out.push_str(&text[last..c.start()]);
            out.push_str(&repl(&c));
            last = c.end();
        }

        out.push_str(&text[last..]);
        Ok(out)
    }

    /// `re.sub` with a fixed replacement string (no group references)
    pub fn sub(&self, text: &str, repl: &str) -> Result<String> {
        self.sub_with(text, |_| repl.to_owned())
    }

    /// `re.sub` whose replacement only fails as the Python function would raise
    pub fn try_sub_with(
        &self,
        text: &str,
        mut repl: impl FnMut(&Caps) -> Result<String>,
    ) -> Result<String> {
        let mut out = String::with_capacity(text.len());
        let mut last = 0;
        for c in self.captures_all(text)? {
            out.push_str(&text[last..c.start()]);
            out.push_str(&repl(&c)?);
            last = c.end();
        }

        out.push_str(&text[last..]);
        Ok(out)
    }

    /// Whether the pattern matches anywhere
    pub fn is_match(&self, text: &str) -> Result<bool> {
        Ok(match &self.0 {
            Engine::Plain(re) => re.is_match(text),
            Engine::Fancy(re) => re.is_match(text).wrap_err("regex match")?,
        })
    }
}

/// A pattern compiled on first use
pub struct Lazy(std::sync::LazyLock<PyRe>);

impl Lazy {
    pub const fn new(init: fn() -> PyRe) -> Self {
        Self(std::sync::LazyLock::new(init))
    }
}

impl std::ops::Deref for Lazy {
    type Target = PyRe;

    fn deref(&self) -> &PyRe {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_classes_inside_and_outside_brackets() {
        let re = PyRe::plain(r"[^\s<]+\d");
        let caps = re.search("\u{1c}ab1").unwrap().unwrap();
        assert_eq!(caps.as_str(), "ab1");
        let re = PyRe::plain(r"<\|([\d.]+)\|>");
        assert_eq!(re.search("<|١.5|>").unwrap().unwrap().group(1), "١.5");
    }
}
