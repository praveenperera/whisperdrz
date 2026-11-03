//! `DecoderState` (`decoding.py:297-848`): the vocab, token sets and bool masks the
//! constraint state machine reads, built once per tokenizer

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::{Arc, Mutex};

use eyre::{Result, eyre};

use super::mask::TokenMask;
use crate::text::vocab::starts_with_timing;
use crate::text::{Token, TokenLists, Tokenizer, Vocab, py};

/// `VALID_BRACKET_TAGS`: the only `[tag]` bodies the decoder lets through
pub const VALID_BRACKET_TAGS: [&str; 93] = [
    // body sounds
    "breath",
    "cough",
    "sniff",
    "sigh",
    "gasp",
    "clearthroat",
    "swallow",
    "sneeze",
    "wheeze",
    "exhale",
    "noseexhale",
    "sharpinhale",
    // mouth sounds
    "tut",
    "click",
    "kiss",
    "kissing",
    "kissteeth",
    "chew",
    "chewing",
    "clap",
    "clapping",
    "snap",
    "slurp",
    "mouthnoise",
    // laugh variants
    "laugh",
    "chuckle",
    "giggle",
    "giggling",
    "scoff",
    "laughter",
    // speech
    "hesitate",
    "mumble",
    "inaudible",
    "babble",
    "slurring",
    "mispeak",
    "lisp",
    // crowd
    "applause",
    "cheering",
    "crowdnoise",
    "crowdlaugh",
    "crowdlaughing",
    "crowdcheering",
    "crowdchatter",
    "crowdclapping",
    "crowdcheer",
    "crowdcough",
    "crowdgasp",
    "jeering",
    // actions
    "pause",
    "panting",
    "crying",
    "screaming",
    "scream",
    "barking",
    "grunt",
    "whimper",
    "whistle",
    "squeal",
    "yelp",
    "cooing",
    "imitating",
    "struggling",
    // paired opening
    "laughing",
    "singing",
    "whisper",
    "shouting",
    "music",
    "voicebreaking",
    "shout",
    "sing",
    // paired closing
    "/laughing",
    "/singing",
    "/whisper",
    "/shouting",
    "/music",
    "/voicebreaking",
    "/shout",
    "/sing",
    "/screaming",
    "/mispeak",
    "/slurring",
    "/clapping",
    "/crowdcheering",
    // stop variants
    "stoplaughing",
    "stopmusic",
    "stopshouting",
    "stopsinging",
    "stopmispeak",
    "stopwhisper",
    "stopscreaming",
    "stopvoicebreaking",
    // special
    "c",
];

/// `0.02`: one timing token step in seconds
pub const TIMING_STEP: f64 = 0.02;
/// Index of the last timing token (`<|30.00|>`)
pub const MAX_TIMING_IDX: usize = 1500;

/// The `][` token, blocked on every step
pub const BRACKET_CLOSE_OPEN: Token = 7131;

/// Single-character token ids (`DecoderState._single_char_ids`)
#[derive(Debug, Clone, Copy)]
pub struct CharIds {
    pub open_bracket: Token,
    pub close_bracket: Token,
    pub open_paren: Token,
    pub close_paren: Token,
    pub less: Token,
    pub greater: Token,
    pub space: Token,
    pub zero: Token,
    pub newline: Token,
    pub asterisk: Token,
}

/// The bool masks of `init_gpu_tensors`, by their python cache names
pub struct Masks {
    pub base_blocked: TokenMask,
    pub base_extra_blocked: TokenMask,
    pub angle_blocked_mask: TokenMask,
    pub asterisk_blocked_mask: TokenMask,
    pub block_open_bracket: TokenMask,
    pub block_close_bracket: TokenMask,
    pub block_open_paren: TokenMask,
    pub block_close_paren: TokenMask,
    pub block_less_than: TokenMask,
    pub block_greater_than: TokenMask,
    pub block_angle_consecutive: TokenMask,
    pub block_asterisk: TokenMask,
    pub space_asterisk_token: TokenMask,
    pub block_contains_space: TokenMask,
    pub block_contains_close_bracket: TokenMask,
    pub block_contains_open_bracket: TokenMask,
    pub block_quote: TokenMask,
    pub block_timings: TokenMask,
    pub block_space_starting: TokenMask,
    pub block_digit_starting: TokenMask,
    pub block_punct: TokenMask,
    pub block_newlines: TokenMask,
    pub in_bracket_blocked: TokenMask,
    pub in_angle_blocked: TokenMask,
    pub in_paren_blocked: TokenMask,
    pub after_timing_blocked: TokenMask,
    pub lang_enabled: TokenMask,
    pub space_starting_enabled: TokenMask,
    pub speaker_id_mask: TokenMask,
}

/// Port of `DecoderState`: built from the tokenizer, read-only afterwards except for
/// the per-language base mask cache
pub struct DecoderTables {
    pub tok: Tokenizer,
    pub vocab: Vocab,
    /// The `inference_utils` lists (`dont_predict` and friends)
    pub lists: TokenLists,
    /// `timings`: timing token ids in vocab order
    pub timings: Vec<Token>,
    timing_set: HashSet<Token>,
    /// `timing_token_ids`: timing ids sorted by their value, index `k` is `k * 0.02` s
    pub timing_token_ids: Vec<Token>,
    pub space_starting: Vec<Token>,
    pub digit_starting: Vec<Token>,
    pub space_tokens_only: Vec<Token>,
    pub punctuation: Vec<Token>,
    /// `_punct_block_sets` for `! ? , - . … '`
    pub punct_block_sets: Vec<Vec<Token>>,
    /// `_all_punct_blocked`
    pub all_punct: BTreeSet<Token>,
    /// `_bracket_candidates`: ids that could sit inside a `[tag]`
    pub bracket_candidates: Vec<Token>,
    /// `bracket_allowed`: for every valid tag prefix (and `""`), the tokens that keep
    /// it a valid prefix in ascending id order, then `]` when the prefix is a whole tag
    pub bracket_allowed: HashMap<String, Vec<Token>>,
    /// `VALID_BRACKET_PREFIXES`
    pub bracket_prefixes: HashSet<String>,
    pub ids: CharIds,
    pub masks: Masks,
    base_by_lang: Mutex<HashMap<String, Arc<TokenMask>>>,
}

impl DecoderTables {
    pub fn new(tok: Tokenizer) -> Result<Self> {
        let vocab = Vocab::new(&tok)?;
        let lists = TokenLists::new(&vocab, &tok)?;
        let v = vocab.len();
        let first = |text: &str| -> Result<Token> {
            tok.encode(text)?
                .first()
                .copied()
                .ok_or_else(|| eyre!("{text:?} encodes to nothing"))
        };

        let timings: Vec<Token> = vocab
            .iter()
            .filter(|(_, t)| starts_with_timing(t))
            .map(|(i, _)| i)
            .collect();
        let timing_set: HashSet<Token> = timings.iter().copied().collect();

        // `timings_values = {parse_timing(text): id}` sorted by value; a repeated value
        // keeps the later id, as the dict comprehension does
        let mut by_value: Vec<(f64, Token)> = Vec::new();
        for &t in &timings {
            let value = parse_timing(vocab.text(t)).ok_or_else(|| eyre!("bad timing token {t}"))?;
            match by_value.iter_mut().find(|(x, _)| *x == value) {
                Some(slot) => slot.1 = t,
                None => by_value.push((value, t)),
            }
        }

        by_value.sort_by(|a, b| a.0.total_cmp(&b.0));
        let timing_token_ids: Vec<Token> = by_value.into_iter().map(|(_, t)| t).collect();

        let space_starting: Vec<Token> = vocab
            .iter()
            .filter(|(_, t)| t.starts_with(' '))
            .map(|(i, _)| i)
            .collect();
        let digit_starting: Vec<Token> = vocab
            .iter()
            .filter(|(_, t)| t.chars().next().is_some_and(py::is_digit))
            .map(|(i, _)| i)
            .collect();
        let space_tokens_only: Vec<Token> = vocab
            .iter()
            .filter(|(_, t)| {
                let mut c = t.chars();
                c.next() == Some(' ') && c.next().is_some_and(py::is_alnum)
            })
            .map(|(i, _)| i)
            .collect();

        let punctuation: Vec<Token> = vocab
            .revocab_items()
            .filter(|(t, i)| {
                let s = py::lstrip(t);
                ["!", "?", ".", ",", "-"].iter().any(|p| s.starts_with(p))
                    && !timing_set.contains(i)
            })
            .map(|(_, i)| i)
            .collect();

        let punct_block_sets: Vec<Vec<Token>> = ["!", "?", ",", "-", ".", "…", "'"]
            .iter()
            .map(|p| {
                vocab
                    .containing(p)
                    .into_iter()
                    .filter(|t| !timing_set.contains(t))
                    .collect()
            })
            .collect();
        let mut all_punct: BTreeSet<Token> = punctuation.iter().copied().collect();
        for set in &punct_block_sets {
            all_punct.extend(set);
        }

        const BAD: &str = " \t\n[]()<>|.,!?-'\"*";
        let bracket_candidates: Vec<Token> = vocab
            .iter()
            .filter(|(_, t)| {
                !t.is_empty() && !t.chars().any(|c| BAD.contains(c) || py::is_digit(c))
            })
            .map(|(i, _)| i)
            .collect();

        let ids = CharIds {
            open_bracket: first("[")?,
            close_bracket: first("]")?,
            open_paren: first("(")?,
            close_paren: first(")")?,
            less: first("<")?,
            greater: first(">")?,
            space: first(" ")?,
            zero: first("0")?,
            newline: first("\n")?,
            asterisk: first("*")?,
        };

        let mut bracket_prefixes = HashSet::new();
        for tag in VALID_BRACKET_TAGS {
            for (end, _) in tag.char_indices().skip(1).chain([(tag.len(), ' ')]) {
                bracket_prefixes.insert(tag[..end].to_owned());
            }
        }

        // python tests every candidate against every prefix; looking the remainder of
        // each longer prefix up by text gives the same sets, in ascending id order
        let mut candidates_by_text: HashMap<&str, Vec<Token>> = HashMap::new();
        for &t in &bracket_candidates {
            candidates_by_text.entry(vocab.text(t)).or_default().push(t);
        }

        let mut bracket_allowed = HashMap::new();
        for prefix in bracket_prefixes.iter().map(String::as_str).chain([""]) {
            let mut allowed: Vec<Token> = bracket_prefixes
                .iter()
                .filter(|q| q.len() > prefix.len() && q.starts_with(prefix))
                .flat_map(|q| {
                    candidates_by_text
                        .get(&q[prefix.len()..])
                        .into_iter()
                        .flatten()
                        .copied()
                })
                .collect();
            allowed.sort_unstable();
            if VALID_BRACKET_TAGS.contains(&prefix) {
                allowed.push(ids.close_bracket);
            }

            bracket_allowed.insert(prefix.to_owned(), allowed);
        }

        let mask = |ids: &[Token]| TokenMask::from_ids(v, ids.iter().copied());
        let one = |id: Token| TokenMask::from_ids(v, [id]);
        let containing = |c: &str| mask(&vocab.containing(c));

        // space+punct tokens and `][` are blocked for every language
        let space_punct: Vec<Token> = vocab
            .revocab_items()
            .filter(|(t, _)| {
                let mut c = t.chars();
                c.next() == Some(' ') && c.next().is_some_and(|c| "?!.,-'\"…".contains(c))
            })
            .map(|(_, i)| i)
            .collect();
        let mut base_extra_blocked = mask(&space_punct);
        base_extra_blocked.set(BRACKET_CLOSE_OPEN);
        let mut base_blocked = mask(&lists.dont_predict);
        base_blocked.or(&base_extra_blocked);

        let block_timings = mask(&timings);
        let block_punct = TokenMask::from_ids(v, all_punct.iter().copied());
        let block_contains_space = containing(" ");
        let block_digit_starting = mask(&digit_starting);
        let block_open_bracket = one(ids.open_bracket);
        let block_open_paren = one(ids.open_paren);
        let block_less_than = one(ids.less);
        let union = |parts: &[&TokenMask]| {
            let mut m = TokenMask::new(v);
            for p in parts {
                m.or(p);
            }

            m
        };

        let in_bracket_blocked = union(&[
            &block_contains_space,
            &block_punct,
            &block_open_bracket,
            &block_timings,
            &block_digit_starting,
        ]);
        let in_angle_blocked = union(&[
            &block_contains_space,
            &block_punct,
            &block_less_than,
            &block_timings,
            &block_digit_starting,
        ]);
        let in_paren_blocked = union(&[
            &block_contains_space,
            &block_punct,
            &block_open_paren,
            &block_timings,
        ]);
        let after_timing_blocked = union(&[&block_timings, &block_punct, &block_open_paren]);

        let mut space_starting_enabled = mask(&space_starting);
        space_starting_enabled.set(ids.newline);

        let mut speaker_id_mask = TokenMask::new(v);
        for ch in "0123456789c".chars() {
            speaker_id_mask.set(first(&ch.to_string())?);
        }

        let masks = Masks {
            angle_blocked_mask: mask(&lists.angle_tokens_blocked),
            asterisk_blocked_mask: mask(&lists.asterisk_tokens_unblockable),
            block_close_bracket: one(ids.close_bracket),
            block_close_paren: one(ids.close_paren),
            block_greater_than: one(ids.greater),
            block_angle_consecutive: containing("><"),
            block_asterisk: one(ids.asterisk),
            space_asterisk_token: one(first(" *")?),
            block_contains_close_bracket: containing("]"),
            block_contains_open_bracket: containing("["),
            block_quote: containing("\""),
            block_space_starting: mask(&space_starting),
            block_newlines: one(ids.newline),
            lang_enabled: mask(&tok.all_language_tokens()),
            base_blocked,
            base_extra_blocked,
            block_open_bracket,
            block_open_paren,
            block_less_than,
            block_contains_space,
            block_timings,
            block_digit_starting,
            block_punct,
            in_bracket_blocked,
            in_angle_blocked,
            in_paren_blocked,
            after_timing_blocked,
            space_starting_enabled,
            speaker_id_mask,
        };

        let base_by_lang = Mutex::new(HashMap::from([(
            "en".to_owned(),
            Arc::new(masks.base_blocked.clone()),
        )]));
        Ok(Self {
            tok,
            vocab,
            lists,
            timings,
            timing_set,
            timing_token_ids,
            space_starting,
            digit_starting,
            space_tokens_only,
            punctuation,
            punct_block_sets,
            all_punct,
            bracket_candidates,
            bracket_allowed,
            bracket_prefixes,
            ids,
            masks,
            base_by_lang,
        })
    }

    /// Vocab size `V`
    pub fn n_vocab(&self) -> usize {
        self.vocab.len()
    }

    pub fn is_timing(&self, id: Token) -> bool {
        self.timing_set.contains(&id)
    }

    /// `get_base_blocked_cpu(lang)`: `dont_predict` with the language's script block,
    /// plus the language-independent extras. Empty and `"auto"` mean English
    pub fn base_blocked(&self, language: &str) -> Arc<TokenMask> {
        let key = if language.is_empty() || language == "auto" {
            "en"
        } else {
            language
        };
        let mut cache = self.base_by_lang.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(m) = cache.get(key) {
            return m.clone();
        }

        let dp = self.lists.dont_predict_for_lang(&self.vocab, key);
        let mut m = TokenMask::from_ids(self.n_vocab(), dp);
        m.or(&self.masks.base_extra_blocked);
        let m = Arc::new(m);
        cache.insert(key.to_owned(), m.clone());
        m
    }

    /// `timing_token_ids[start..end]`, clamped to the table
    pub fn timing_ids(&self, start: usize, end: usize) -> &[Token] {
        let end = end.min(self.timing_token_ids.len());
        &self.timing_token_ids[start.min(end)..end]
    }
}

/// `clean.parse_timing`: `float` of the text with `<|` and `|>` removed
pub fn parse_timing(t: &str) -> Option<f64> {
    py::float(&t.replace("<|", "").replace("|>", ""))
}
