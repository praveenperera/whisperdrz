//! Decoded vocabulary, script-based token blocking (`vocab_constraints.py`) and the
//! blocked-token lists `inference_utils._init_token_lists` builds
//!
//! Lists come back in Python's order where Python builds a list. Where Python goes
//! through a set (`never_predict`, `get_always_blocked_tokens`), the order is CPython's
//! hash order, which nothing downstream reads: every consumer turns them into masks

use std::collections::{BTreeSet, HashMap, HashSet};

use eyre::{Result, eyre};

use super::py;
use super::tokenizer::{Token, Tokenizer};

/// `VOCAB`/`ds.vocab_list`: every token decoded on its own, plus the reverse map
pub struct Vocab {
    tokens: Vec<String>,
    /// `REVOCAB`: string to the last id that decodes to it (later ids overwrite earlier
    /// ones in python's dict comprehension)
    revocab: HashMap<String, Token>,
    /// Distinct strings in first-occurrence order, which is `REVOCAB`'s key order
    revocab_order: Vec<String>,
}

impl Vocab {
    pub fn new(tok: &Tokenizer) -> Result<Self> {
        let tokens = (0..tok.n_vocab() as Token)
            .map(|i| tok.decode(&[i]))
            .collect::<Result<Vec<_>>>()?;
        let mut revocab = HashMap::new();
        let mut revocab_order = Vec::new();
        for (i, t) in tokens.iter().enumerate() {
            if revocab.insert(t.clone(), i as Token).is_none() {
                revocab_order.push(t.clone());
            }
        }

        Ok(Self {
            tokens,
            revocab,
            revocab_order,
        })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// The decoded text of one token
    pub fn text(&self, id: Token) -> &str {
        &self.tokens[id as usize]
    }

    pub fn iter(&self) -> impl Iterator<Item = (Token, &str)> {
        self.tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (i as Token, t.as_str()))
    }

    /// `REVOCAB.items()` in python's order: one id per distinct string, the last one
    pub fn revocab_items(&self) -> impl Iterator<Item = (&str, Token)> {
        self.revocab_order
            .iter()
            .map(|t| (t.as_str(), self.revocab[t]))
    }

    pub fn revocab(&self, text: &str) -> Option<Token> {
        self.revocab.get(text).copied()
    }

    /// `tokens_with_c(t)` / `DecoderState.tokens_containing(t)`: ids whose text contains `t`
    pub fn containing(&self, t: &str) -> Vec<Token> {
        self.iter()
            .filter(|(_, s)| s.contains(t))
            .map(|(i, _)| i)
            .collect()
    }

    /// `TIMINGS`: ids whose text starts with a `<|d.d|>` timestamp
    pub fn timings(&self) -> Vec<Token> {
        self.iter()
            .filter(|(_, s)| starts_with_timing(s))
            .map(|(i, _)| i)
            .collect()
    }
}

/// `re.match(TIMINGS_FORMAT, s)`: `<|\d+\.\d+|>` at the start of `s`
pub fn starts_with_timing(s: &str) -> bool {
    timing_prefix_len(s).is_some()
}

/// Byte length of a leading `<|\d+\.\d+|>`, digits being any Unicode Nd as in python's `\d`
pub fn timing_prefix_len(s: &str) -> Option<usize> {
    let rest = s.strip_prefix("<|")?;
    let digits = |r: &str| {
        r.char_indices()
            .find(|&(_, c)| !py::is_decimal(c))
            .map_or(r.len(), |(i, _)| i)
    };
    let a = digits(rest);
    if a == 0 {
        return None;
    }

    let rest2 = rest[a..].strip_prefix('.')?;
    let b = digits(rest2);
    if b == 0 {
        return None;
    }

    rest2[b..].strip_prefix("|>")?;
    Some(2 + a + 1 + b + 2)
}

/// Scripts named by `_get_token_script`
const SCRIPTS: [&str; 22] = [
    "LATIN",
    "CYRILLIC",
    "ARABIC",
    "HEBREW",
    "GREEK",
    "DEVANAGARI",
    "BENGALI",
    "TAMIL",
    "TELUGU",
    "GUJARATI",
    "GURMUKHI",
    "KANNADA",
    "MALAYALAM",
    "ORIYA",
    "SINHALA",
    "THAI",
    "LAO",
    "MYANMAR",
    "ARMENIAN",
    "GEORGIAN",
    "ETHIOPIC",
    "CJK",
];

/// `_get_token_script`: the script of the first character that names one, from its
/// Unicode character name
pub fn token_script(token: &str) -> Option<&'static str> {
    for c in token.chars() {
        if c == ' ' || c == '\n' || c == '\t' || (c.is_ascii() && !c.is_ascii_alphabetic()) {
            continue;
        }

        if c.is_ascii() {
            return Some("LATIN");
        }

        let Some(name) = unicode_names2::name(c) else {
            continue;
        };
        let name = name.to_string();
        if name.contains("HIRAGANA") {
            return Some("HIRAGANA");
        }

        if name.contains("KATAKANA") {
            return Some("KATAKANA");
        }

        if name.contains("HANGUL") {
            return Some("HANGUL");
        }

        if name.contains("IDEOGRAPHIC") {
            return Some("CJK");
        }

        let first = name.split(' ').next().unwrap_or("");
        if let Some(s) = SCRIPTS.iter().find(|s| **s == first) {
            return Some(s);
        }
    }

    None
}

/// `_get_language_group(code)["allowed_scripts"]`; unknown codes get Latin only
pub fn allowed_scripts(language: &str) -> &'static [&'static str] {
    const GROUPS: &[(&[&str], &[&str])] = &[
        (
            &[
                "en", "es", "fr", "it", "pt", "de", "nl", "sv", "no", "da", "fi", "pl", "cs", "sk",
                "ro", "hu", "hr", "sl", "sq", "et", "lv", "lt", "tr", "id", "ms", "tl", "vi", "sw",
                "so", "af", "eu", "ca", "gl", "is", "ga", "cy", "mt", "lb",
            ],
            &["LATIN"],
        ),
        (
            &[
                "ru", "uk", "bg", "sr", "bs", "mk", "be", "kk", "ky", "mn", "uz", "tg",
            ],
            &["LATIN", "CYRILLIC"],
        ),
        (&["el"], &["LATIN", "GREEK"]),
        (&["ar", "fa", "ur", "ps"], &["LATIN", "ARABIC"]),
        (&["he"], &["LATIN", "HEBREW"]),
        (
            &["zh", "ja", "ko", "yue"],
            &["LATIN", "CJK", "HIRAGANA", "KATAKANA", "HANGUL"],
        ),
        (
            &[
                "hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as", "ur", "ne", "si",
            ],
            &[
                "LATIN",
                "DEVANAGARI",
                "BENGALI",
                "TAMIL",
                "TELUGU",
                "GUJARATI",
                "GURMUKHI",
                "KANNADA",
                "MALAYALAM",
                "ORIYA",
                "SINHALA",
            ],
        ),
        (&["th"], &["LATIN", "THAI"]),
        (&["hy"], &["LATIN", "ARMENIAN"]),
        (&["ka"], &["LATIN", "GEORGIAN"]),
        (&["am", "ti"], &["LATIN", "ETHIOPIC"]),
    ];

    // first group wins, as in python's dict scan ("ur" is in both arabic and indic)
    GROUPS
        .iter()
        .find(|(langs, _)| langs.contains(&language))
        .map_or(&["LATIN"], |(_, scripts)| scripts)
}

/// `get_blocked_tokens_for_language(code)`: ordinary tokens whose script the
/// language doesn't allow, in id order
pub fn blocked_for_language(vocab: &Vocab, language: &str) -> Vec<Token> {
    if language.is_empty() || language == "auto" {
        return Vec::new();
    }

    let allowed = allowed_scripts(&language.to_lowercase());
    vocab
        .iter()
        .filter(|(_, t)| !t.starts_with("<|"))
        .filter(|(_, t)| token_script(t).is_some_and(|s| !allowed.contains(&s)))
        .map(|(i, _)| i)
        .collect()
}

/// `get_always_blocked_tokens()` as a sorted set (python returns `tuple(set(...))`)
pub fn always_blocked(vocab: &Vocab, tok: &Tokenizer) -> BTreeSet<Token> {
    const CHARS: [&str; 21] = [
        "\u{fffd}", "/", "{", "}", "+", "=", "^", "~", "_", "#", "»", "«", ";", ":", "《", "》",
        "「", "」", "『", "』", "؟",
    ];

    let mut out = BTreeSet::new();
    for c in CHARS {
        out.extend(vocab.containing(c));
    }

    // python's try block stops at the first of these that isn't a single token
    for c in ["<", ">", "|"] {
        match tok.single_token(c) {
            Some(id) => {
                out.insert(id);
            }
            None => break,
        }
    }

    out
}

/// The module-level token lists `init_for_model` builds in `inference_utils`
pub struct TokenLists {
    /// `TIMINGS`
    pub timings: Vec<Token>,
    /// `dont_predict` (with duplicates, as python's list has them)
    pub dont_predict: Vec<Token>,
    /// `angle_tokens_blocked`
    pub angle_tokens_blocked: Vec<Token>,
    /// `asterisk_tokens_unblockable`
    pub asterisk_tokens_unblockable: Vec<Token>,
    /// `initial_tokens`
    pub initial_tokens: Vec<Token>,
}

/// Token the model hallucinates as "Kivol" (`inference_utils`, blocked by id)
const KIVOL: Token = 27506;

impl TokenLists {
    pub fn new(vocab: &Vocab, tok: &Tokenizer) -> Result<Self> {
        let timings = vocab.timings();
        let timing_set: HashSet<Token> = timings.iter().copied().collect();
        let first = |text: &str| -> Result<Token> {
            tok.encode(text)?
                .first()
                .copied()
                .ok_or_else(|| eyre!("{text:?} encodes to nothing"))
        };

        let mut never_predict: BTreeSet<Token> =
            blocked_for_language(vocab, "en").into_iter().collect();
        never_predict.extend(always_blocked(vocab, tok));
        let mut never_predict: Vec<Token> = never_predict.into_iter().collect();

        let space_punctuation: Vec<Token> = vocab
            .revocab_items()
            .filter(|(t, _)| {
                let mut c = t.chars();
                c.next() == Some(' ') && c.next().is_some_and(|c| "!?.,-".contains(c))
            })
            .map(|(_, i)| i)
            .collect();

        let double_punctuation: Vec<Token> = vocab
            .revocab_items()
            .filter(|(t, _)| {
                ["!!", "??", "..", ",,", "--", "''"]
                    .iter()
                    .any(|d| t.contains(d))
                    && !t.contains("...")
            })
            .map(|(_, i)| i)
            .collect();

        never_predict.extend(
            vocab
                .iter()
                .filter(|(i, t)| {
                    t.contains("<|") && !t.contains("endoftext|>") && !timing_set.contains(i)
                })
                .map(|(i, _)| i),
        );

        let initial_tokens = vec![tok.sot()];

        let eot = tok.eot();
        let angle_ok =
            |&i: &Token| !timing_set.contains(&i) && i != eot && !vocab.text(i).contains("><");
        let mut angle_tokens_blocked: Vec<Token> =
            vocab.containing("<").into_iter().filter(angle_ok).collect();
        angle_tokens_blocked.extend(vocab.containing(">").into_iter().filter(angle_ok));

        // python builds `{encode("*")[0], encode(" *")[0]}`; listed in ascending order
        // here, which is also CPython's iteration order for these two small ints
        let mut asterisk_tokens_unblockable = vec![first("*")?, first(" *")?];
        asterisk_tokens_unblockable.sort_unstable();
        asterisk_tokens_unblockable.dedup();
        let asterisk_blocked: Vec<Token> = vocab
            .containing("*")
            .into_iter()
            .filter(|t| !asterisk_tokens_unblockable.contains(t))
            .collect();

        let mut dont_predict = never_predict;
        dont_predict.extend(&initial_tokens);
        dont_predict.extend(double_punctuation);
        dont_predict.extend(space_punctuation);
        dont_predict.push(tok.no_timestamps());
        dont_predict.extend(asterisk_blocked);
        dont_predict.extend(&asterisk_tokens_unblockable);
        for c in ["(", ")", " ]"] {
            dont_predict.extend(vocab.containing(c));
        }

        dont_predict.extend(&angle_tokens_blocked);
        for c in ["【", "】", "♪", "一", "]["] {
            dont_predict.extend(vocab.containing(c));
        }

        dont_predict.push(KIVOL);
        // the python list also has a "smart quotes" entry whose literal is a broken
        // triple-quoted string spanning two lines; it matches no token, so it's left out
        for c in ["[...]", "[?", "–", "—", "\u{2018}", "\u{2019}", "´"] {
            dont_predict.extend(vocab.containing(c));
        }

        Ok(Self {
            timings,
            dont_predict,
            angle_tokens_blocked,
            asterisk_tokens_unblockable,
            initial_tokens,
        })
    }

    /// `get_dont_predict_for_lang(lang)` as a sorted set: `dont_predict` with the
    /// English script block swapped for the language's
    pub fn dont_predict_for_lang(&self, vocab: &Vocab, language: &str) -> BTreeSet<Token> {
        let mut dp: BTreeSet<Token> = self.dont_predict.iter().copied().collect();
        if language.is_empty() || language == "auto" {
            return dp;
        }

        for t in blocked_for_language(vocab, "en") {
            dp.remove(&t);
        }

        dp.extend(blocked_for_language(vocab, language));
        dp
    }
}
