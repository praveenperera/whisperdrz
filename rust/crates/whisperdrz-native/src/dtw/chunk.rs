//! Char-level tokenization, speaker-aware chunking and per-chunk prep
//! (`tokens_chars`, `_render_chunk`, `chunk_speaker_lines`, `_prepare_chunk`)

use std::collections::HashMap;

use super::DtwTokenizer;
use super::kernel::WordAnchor;
use super::parse::{PredWord, Speaker, SpeakerLine, TagAnchor, is_py_space};

/// `text.split()`: runs of Python whitespace separate, empty pieces dropped
fn py_split(text: &str) -> impl Iterator<Item = &str> {
    text.split(is_py_space).filter(|w| !w.is_empty())
}

/// A tokenizer with a per-char cache, as `_char_tokens`
pub struct CharTokens<'t, T: DtwTokenizer + ?Sized> {
    tokenizer: &'t T,
    cache: HashMap<char, Vec<u32>>,
}

/// Char-level tokens of a rendered chunk
#[derive(Debug, Clone, PartialEq)]
pub struct TokensChars {
    /// Every text token, a space token before each word but the first
    pub ids: Vec<u32>,
    /// The words, then `""` for the eot group
    pub words: Vec<String>,
    /// Each word's tokens (with its leading space), then `[eot]`
    pub word_tokens: Vec<Vec<u32>>,
}

impl<'t, T: DtwTokenizer + ?Sized> CharTokens<'t, T> {
    pub fn new(tokenizer: &'t T) -> Self {
        Self {
            tokenizer,
            cache: HashMap::new(),
        }
    }

    pub fn tokenizer(&self) -> &'t T {
        self.tokenizer
    }

    /// `encode(ch)` without special tokens
    fn char_tokens(&mut self, ch: char) -> &[u32] {
        let tokenizer = self.tokenizer;
        self.cache.entry(ch).or_insert_with(|| {
            let eot = tokenizer.eot();
            tokenizer
                .encode(ch.encode_utf8(&mut [0; 4]))
                .into_iter()
                .filter(|&t| t < eot)
                .collect()
        })
    }

    /// Char-level tokenization with an explicit space token between words, as `tokens_chars`
    ///
    /// The eot sits in its own trailing group so DTW keeps the last word's timing
    pub fn tokens_chars(&mut self, plain_text: &str) -> TokensChars {
        let words: Vec<String> = py_split(plain_text).map(str::to_owned).collect();
        if words.is_empty() {
            return TokensChars {
                ids: Vec::new(),
                words: Vec::new(),
                word_tokens: Vec::new(),
            };
        }

        let space = self.char_tokens(' ').to_vec();
        let mut ids = Vec::new();
        let mut word_tokens = Vec::with_capacity(words.len() + 1);
        for (wi, w) in words.iter().enumerate() {
            let start = ids.len();
            if wi > 0 {
                ids.extend_from_slice(&space);
            }

            for ch in w.chars() {
                ids.extend_from_slice(self.char_tokens(ch));
            }

            word_tokens.push(ids[start..].to_vec());
        }

        let mut words = words;
        words.push(String::new());
        word_tokens.push(vec![self.tokenizer.eot()]);
        TokensChars {
            ids,
            words,
            word_tokens,
        }
    }

    /// `len(tokens_chars(text).ids)` without building the groups
    fn count(&mut self, plain_text: &str) -> usize {
        let mut n = 0;
        for (wi, w) in py_split(plain_text).enumerate() {
            if wi > 0 {
                n += self.char_tokens(' ').len();
            }

            for ch in w.chars() {
                n += self.char_tokens(ch).len();
            }
        }

        n
    }
}

/// A speaker line, or the part of one, that went into a chunk
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkLine<'a> {
    pub speaker: &'a Speaker,
    /// Indices into the window's flat word list
    pub words: &'a [usize],
}

/// Renders a chunk with its speakers renumbered from 0 in order of appearance, as
/// `_render_chunk`
///
/// Returns the text and, per space-separated item, the flat word index it shows
/// (`None` for the speaker tags)
pub fn render_chunk(chunk: &[ChunkLine<'_>], words: &[PredWord]) -> (String, Vec<Option<usize>>) {
    let mut speakers: Vec<&Speaker> = Vec::new();
    let mut text = String::new();
    let mut tagged = Vec::new();
    for line in chunk {
        let ns = match speakers.iter().position(|s| *s == line.speaker) {
            Some(k) => k,
            None => {
                speakers.push(line.speaker);
                speakers.len() - 1
            }
        };

        if !text.is_empty() {
            text.push(' ');
        }

        text.push_str(&format!("[{ns}]"));
        tagged.push(None);
        for &wi in line.words {
            text.push(' ');
            text.push_str(&words[wi].text);
            tagged.push(Some(wi));
        }
    }

    (text, tagged)
}

/// Packs speaker lines into chunks of at most `max_tokens` char-level tokens, as
/// `chunk_speaker_lines`
///
/// Lines stay whole where they fit; a line too long on its own is split at the
/// longest prefix that fits (at least one word)
pub fn chunk_speaker_lines<'a, T: DtwTokenizer + ?Sized>(
    lines: &'a [SpeakerLine],
    words: &[PredWord],
    tokens: &mut CharTokens<'_, T>,
    max_tokens: usize,
) -> Vec<Vec<ChunkLine<'a>>> {
    let mut fits =
        |chunk: &[ChunkLine<'_>]| tokens.count(&render_chunk(chunk, words).0) <= max_tokens;

    let mut chunks = Vec::new();
    let mut cur: Vec<ChunkLine<'a>> = Vec::new();
    let mut queue = lines.iter().map(|l| ChunkLine {
        speaker: &l.speaker,
        words: &l.words,
    });
    let mut head = queue.next();
    while let Some(line) = head {
        let mut candidate = cur.clone();
        candidate.push(line);
        if fits(&candidate) {
            cur = candidate;
            head = queue.next();
            continue;
        }

        if !cur.is_empty() {
            chunks.push(std::mem::take(&mut cur));
            continue;
        }

        let (mut lo, mut hi) = (1usize, line.words.len());
        let mut best = 1;
        while lo <= hi {
            let mid = (lo + hi) / 2;
            if fits(&[ChunkLine {
                words: &line.words[..mid],
                ..line
            }]) {
                best = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }

        // `best` is at least 1 even when one word alone is over budget; slicing clamps
        // like Python's `ws[:best]` would for a line of fewer words
        let best = best.min(line.words.len());
        chunks.push(vec![ChunkLine {
            words: &line.words[..best],
            ..line
        }]);
        let rest = ChunkLine {
            words: &line.words[best..],
            ..line
        };
        head = if rest.words.is_empty() {
            queue.next()
        } else {
            Some(rest)
        };
    }

    if !cur.is_empty() {
        chunks.push(cur);
    }

    chunks
}

/// Everything one chunk needs for DTW besides the GPU forward (`_prepare_chunk`)
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkPrep {
    pub ids: Vec<u32>,
    pub ext_words: Vec<String>,
    pub ext_word_tokens: Vec<Vec<u32>>,
    /// Per rendered item, its flat word index (`None` for speaker tags)
    pub tagged: Vec<Option<usize>>,
    pub word_anchors: Vec<WordAnchor>,
    /// Tag anchors with indices into `tagged`
    pub tag_anchors: Vec<TagAnchor>,
}

impl ChunkPrep {
    /// `None` when the chunk is outside the token budget or its words don't line up
    pub fn new<T: DtwTokenizer + ?Sized>(
        chunk: &[ChunkLine<'_>],
        tokens: &mut CharTokens<'_, T>,
        words: &[PredWord],
        tag_anchors: &[TagAnchor],
        max_tokens: usize,
    ) -> Option<Self> {
        let (rendered, tagged) = render_chunk(chunk, words);
        let tc = tokens.tokens_chars(&rendered);
        if tc.ids.len() < 3 || tc.ids.len() > max_tokens || tagged.len() + 1 != tc.words.len() {
            return None;
        }

        let mut word_anchors = Vec::new();
        let mut to_local: HashMap<usize, usize> = HashMap::new();
        for (i, oi) in tagged.iter().enumerate() {
            let Some(oi) = *oi else { continue };
            to_local.insert(oi, i);
            if let (Some(start), Some(end)) = (words[oi].start, words[oi].end) {
                word_anchors.push(WordAnchor {
                    word: i,
                    start,
                    end,
                });
            }
        }

        // a tag at a line boundary keeps whichever neighbour landed in this chunk
        let local = |gi: Option<usize>| gi.and_then(|gi| to_local.get(&gi).copied());
        let tag_anchors = tag_anchors
            .iter()
            .map(|a| TagAnchor {
                prev: local(a.prev),
                next: local(a.next),
                ..*a
            })
            .filter(|a| a.prev.is_some() || a.next.is_some())
            .collect();

        Some(Self {
            ids: tc.ids,
            ext_words: tc.words,
            ext_word_tokens: tc.word_tokens,
            tagged,
            word_anchors,
            tag_anchors,
        })
    }
}
