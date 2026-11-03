//! `clean/tags.py`: paired tags (`[singing] ... [/singing]`) and runs of identical
//! non-speech tags

use std::sync::LazyLock;

use eyre::Result;

use super::pattern::{Lazy, PyRe};
use super::py;

/// `PAIRED_TAGS`. Python builds its patterns from a `frozenset`, whose iteration
/// order varies with the string hash; that order can't reach the output (the
/// alternation is anchored by brackets on both sides, and orphan spans are sorted
/// before removal), so this list is alphabetical
const PAIRED_TAGS: [&str; 15] = [
    "accent",
    "crowdcheering",
    "dc",
    "effect",
    "laughing",
    "mispeak",
    "music",
    "reverb",
    "screaming",
    "shout",
    "shouting",
    "sing",
    "singing",
    "voicebreaking",
    "whisper",
];

static MULTI_SPACE: Lazy = Lazy::new(|| PyRe::plain(r"  +"));
static TIMED_TAG: Lazy = Lazy::new(|| PyRe::plain(r"<\|(\d+\.?\d*)\|>\[([^\]]+)\]<\|(\d+\.?\d*)\|>"));
static UNTIMED_RUN: Lazy = Lazy::new(|| PyRe::fancy(r"\[([a-zA-Z][a-zA-Z]*)\][ \t]+\[\1\]"));

/// A literal under `re.IGNORECASE`: each ASCII letter becomes the class of every
/// code point CPython's `sre` matches it with, which adds U+0130 and U+0131 to `i`,
/// U+017F to `s` and U+212A to `k` (Rust's case folding has only the last two)
fn ignorecase(lit: &str) -> String {
    lit.chars()
        .map(|c| match c.to_ascii_lowercase() {
            'i' => r"[iI\x{130}\x{131}]".to_owned(),
            'k' => r"[kK\x{212A}]".to_owned(),
            's' => r"[sS\x{17F}]".to_owned(),
            l if l.is_ascii_lowercase() => format!("[{l}{}]", l.to_ascii_uppercase()),
            _ => regex::escape(&c.to_string()),
        })
        .collect()
}

static OPENING_PAIRED: Lazy = Lazy::new(|| {
    let alts: Vec<String> = PAIRED_TAGS.iter().map(|t| ignorecase(t)).collect();
    PyRe::plain(&format!(r"\[({})\]", alts.join("|")))
});

static CLOSING_PAIRED: LazyLock<Vec<(&'static str, PyRe)>> = LazyLock::new(|| {
    PAIRED_TAGS
        .iter()
        .map(|&tag| {
            let src = format!(
                r"\[(?:{}|{}|/{})\]",
                ignorecase(&format!("stop{tag}")),
                ignorecase(&format!("{tag}stop")),
                ignorecase(tag)
            );
            (tag, PyRe::plain(&src))
        })
        .collect()
});

fn closing_for(tag: &str) -> Option<&'static PyRe> {
    CLOSING_PAIRED.iter().find(|(t, _)| *t == tag).map(|(_, re)| re)
}

/// `remove_orphan_paired_tags`: on each line, drops opening paired tags with no
/// matching closing tag after them and closing tags with no opening tag before
/// them, then collapses double spaces and strips the line
pub fn remove_orphan_paired_tags(text: &str) -> Result<String> {
    let mut out = Vec::new();
    for line in text.split('\n') {
        let openings = OPENING_PAIRED.captures_all(line)?;
        // `tag_name` is `m.group(1).lower()`; a non-ASCII spelling can lower to a
        // name outside the table, which Python skips
        let names: Vec<String> = openings.iter().map(|m| m.group(1).to_lowercase()).collect();
        let mut orphans: Vec<(usize, usize)> = Vec::new();
        let mut used_closings: Vec<usize> = Vec::new();

        for (m, name) in openings.iter().zip(&names) {
            let Some(closing) = closing_for(name) else { continue };
            let mut found = false;
            for cm in closing.captures_all(line)? {
                if cm.start() > m.end() && !used_closings.contains(&cm.start()) {
                    found = true;
                    used_closings.push(cm.start());
                    break;
                }
            }

            if !found {
                orphans.push((m.start(), m.end()));
            }
        }

        for (tag, closing) in CLOSING_PAIRED.iter() {
            for cm in closing.captures_all(line)? {
                if used_closings.contains(&cm.start()) {
                    continue;
                }

                let opening_before = openings.iter().zip(&names).any(|(m, name)| {
                    name == tag && m.end() <= cm.start() && !orphans.contains(&(m.start(), m.end()))
                });
                if !opening_before {
                    orphans.push((cm.start(), cm.end()));
                }
            }
        }

        orphans.sort_unstable_by(|a, b| b.cmp(a));
        let mut line = line.to_owned();
        for (start, end) in orphans {
            line.replace_range(start..end, "");
        }

        let line = MULTI_SPACE.sub(&line, " ")?;
        out.push(py::strip(&line).to_owned());
    }

    Ok(out.join("\n"))
}

/// One left-to-right pass merging `<|a|>[t]<|b|>` `<|c|>[t]<|d|>` (whitespace
/// between) into `<|a|>[t]<|d|>`; returns whether anything merged
fn merge_timed_pass(txt: &str) -> Result<(String, bool)> {
    let matches = TIMED_TAG.captures_all(txt)?;
    if matches.len() < 2 {
        return Ok((txt.to_owned(), false));
    }

    let mut out = String::with_capacity(txt.len());
    let mut last_end = 0;
    let mut changed = false;
    let mut i = 0;
    while i < matches.len() {
        let m1 = &matches[i];
        if let Some(m2) = matches.get(i + 1) {
            let between = &txt[m1.end()..m2.start()];
            if m1.group(2) == m2.group(2) && py::strip(between).is_empty() {
                out.push_str(&txt[last_end..m1.start()]);
                out.push_str(&format!("<|{}|>[{}]<|{}|>", m1.group(1), m1.group(2), m2.group(3)));
                last_end = m2.end();
                i += 2;
                changed = true;
                continue;
            }
        }

        out.push_str(&txt[last_end..m1.end()]);
        last_end = m1.end();
        i += 1;
    }

    out.push_str(&txt[last_end..]);
    Ok((out, changed))
}

/// `consolidate_adjacent_tags`: merges runs of the same timed tag until nothing
/// changes, then collapses `[tag] [tag]` (spaces or tabs between) into `[tag]`
pub fn consolidate_adjacent_tags(text: &str) -> Result<String> {
    let mut text = text.to_owned();
    loop {
        let (next, changed) = merge_timed_pass(&text)?;
        text = next;
        if !changed {
            break;
        }
    }

    loop {
        let next = UNTIMED_RUN.sub_with(&text, |m| format!("[{}]", m.group(1)))?;
        if next == text {
            return Ok(text);
        }

        text = next;
    }
}
