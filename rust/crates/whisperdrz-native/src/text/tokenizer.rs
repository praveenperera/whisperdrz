//! The tiktoken BPE tokenizer over `multilingual.tiktoken` (`tokenizer.py`)
//!
//! The merge loop is tiktoken's `_byte_pair_merge` (lowest rank first, leftmost on
//! ties) and the pre-split pattern runs on `fancy-regex`, which is what tiktoken's own
//! Rust core uses, so piece boundaries match

use std::collections::HashMap;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use eyre::{Result, WrapErr, bail, ensure, eyre};

/// Language codes in Whisper's token order (`tokenizer.LANGUAGES`)
pub const LANGUAGES: [&str; 100] = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
];

/// GPT-2 pre-tokenizer pattern from `get_encoding`
const PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

const MULTILINGUAL: &str =
    include_str!("../../../../../src/whisperdrz/assets/multilingual.tiktoken");

/// A token id
pub type Token = u32;

/// Byte-level BPE with Whisper's special tokens, the `Tokenizer` returned by
/// `get_tokenizer(multilingual=True, num_languages=..., language="en")`
pub struct Tokenizer {
    ranks: HashMap<Vec<u8>, Token>,
    /// Bytes of every token id, ordinary and special
    bytes: Vec<Vec<u8>>,
    specials: HashMap<String, Token>,
    special_re: regex::Regex,
    pattern: fancy_regex::Regex,
    num_languages: usize,
}

impl Tokenizer {
    /// The large-v3 tokenizer: `multilingual.tiktoken` with 100 languages (vocab 51866)
    pub fn multilingual() -> Result<Self> {
        Self::from_tiktoken(MULTILINGUAL, 100)
    }

    /// Builds a tokenizer from tiktoken rank lines (`base64 rank`) with the first
    /// `num_languages` language tokens, as `get_encoding(name, num_languages)` does
    pub fn from_tiktoken(ranks_src: &str, num_languages: usize) -> Result<Self> {
        ensure!(
            num_languages <= LANGUAGES.len(),
            "{num_languages} languages, at most {}",
            LANGUAGES.len()
        );

        let mut ranks = HashMap::new();
        for (i, line) in ranks_src.lines().enumerate() {
            let mut parts = line.split_whitespace();
            let (Some(token), Some(rank)) = (parts.next(), parts.next()) else {
                continue;
            };
            // the asset's rank 50256 is a bare "=", which python's lenient b64decode
            // reads as an empty token
            let token = if token.bytes().all(|b| b == b'=') {
                Vec::new()
            } else {
                STANDARD
                    .decode(token)
                    .wrap_err_with(|| format!("rank line {}", i + 1))?
            };
            let rank: Token = rank
                .parse()
                .wrap_err_with(|| format!("rank line {}", i + 1))?;
            ranks.insert(token, rank);
        }

        let n_ranks = ranks.len();
        let mut bytes = vec![Vec::new(); n_ranks];
        for (token, &rank) in &ranks {
            let slot = bytes
                .get_mut(rank as usize)
                .ok_or_else(|| eyre!("rank {rank} out of range"))?;
            ensure!(slot.is_empty(), "duplicate rank {rank}");
            *slot = token.clone();
        }

        let mut names = vec![
            "<|endoftext|>".to_owned(),
            "<|startoftranscript|>".to_owned(),
        ];
        names.extend(
            LANGUAGES[..num_languages]
                .iter()
                .map(|l| format!("<|{l}|>")),
        );
        names.extend(
            [
                "<|translate|>",
                "<|transcribe|>",
                "<|startoflm|>",
                "<|startofprev|>",
                "<|nospeech|>",
                "<|notimestamps|>",
            ]
            .map(str::to_owned),
        );
        // same f64 product and two-decimal rounding as python's f"<|{i * 0.02:.2f}|>"
        names.extend((0..1501).map(|i| format!("<|{:.2}|>", i as f64 * 0.02)));

        let mut specials = HashMap::new();
        for name in names {
            let id = bytes.len() as Token;
            bytes.push(name.as_bytes().to_vec());
            specials.insert(name, id);
        }

        // no special token is a prefix of another, so alternation order doesn't matter
        let mut alternatives: Vec<&String> = specials.keys().collect();
        alternatives.sort();
        let special_re = regex::Regex::new(
            &alternatives
                .iter()
                .map(|s| regex::escape(s))
                .collect::<Vec<_>>()
                .join("|"),
        )?;

        Ok(Self {
            ranks,
            bytes,
            specials,
            special_re,
            pattern: fancy_regex::Regex::new(PATTERN)?,
            num_languages,
        })
    }

    /// Vocab size including special tokens (`encoding.n_vocab`)
    pub fn n_vocab(&self) -> usize {
        self.bytes.len()
    }

    /// The raw bytes of one token
    pub fn token_bytes(&self, id: Token) -> Option<&[u8]> {
        self.bytes.get(id as usize).map(Vec::as_slice)
    }

    /// `Tokenizer.encode(text)`: `encoding.encode(text, allowed_special="all")`
    pub fn encode(&self, text: &str) -> Result<Vec<Token>> {
        let mut out = Vec::new();
        let mut start = 0;
        for special in self.special_re.find_iter(text) {
            self.encode_ordinary_into(&text[start..special.start()], &mut out)?;
            out.push(self.specials[special.as_str()]);
            start = special.end();
        }

        self.encode_ordinary_into(&text[start..], &mut out)?;
        Ok(out)
    }

    fn encode_ordinary_into(&self, text: &str, out: &mut Vec<Token>) -> Result<()> {
        for piece in self.pattern.find_iter(text) {
            let piece = piece?.as_str().as_bytes();
            match self.ranks.get(piece) {
                Some(&id) => out.push(id),
                None => self.byte_pair_encode(piece, out)?,
            }
        }

        Ok(())
    }

    /// tiktoken's `byte_pair_encode` / `_byte_pair_merge`
    fn byte_pair_encode(&self, piece: &[u8], out: &mut Vec<Token>) -> Result<()> {
        let rank = |range: &[u8]| self.ranks.get(range).copied().unwrap_or(Token::MAX);

        // (start offset, rank of the pair starting here)
        let mut parts: Vec<(usize, Token)> = Vec::with_capacity(piece.len() + 1);
        let mut min = (Token::MAX, usize::MAX);
        for i in 0..piece.len() - 1 {
            let r = rank(&piece[i..i + 2]);
            if r < min.0 {
                min = (r, i);
            }

            parts.push((i, r));
        }

        parts.push((piece.len() - 1, Token::MAX));
        parts.push((piece.len(), Token::MAX));

        let pair_rank = |parts: &[(usize, Token)], i: usize| {
            if i + 3 < parts.len() {
                rank(&piece[parts[i].0..parts[i + 3].0])
            } else {
                Token::MAX
            }
        };
        while min.0 != Token::MAX {
            let i = min.1;
            if i > 0 {
                parts[i - 1].1 = pair_rank(&parts, i - 1);
            }

            parts[i].1 = pair_rank(&parts, i);
            parts.remove(i + 1);

            min = (Token::MAX, usize::MAX);
            for (j, &(_, r)) in parts[..parts.len() - 1].iter().enumerate() {
                if r < min.0 {
                    min = (r, j);
                }
            }
        }

        for pair in parts.windows(2) {
            let bytes = &piece[pair[0].0..pair[1].0];
            match self.ranks.get(bytes) {
                Some(&id) => out.push(id),
                None => bail!("piece {bytes:?} has no rank"),
            }
        }

        Ok(())
    }

    /// `Tokenizer.decode(ids)`: the token bytes joined, decoded as UTF-8 with
    /// `errors="replace"`
    pub fn decode(&self, ids: &[Token]) -> Result<String> {
        let mut joined = Vec::new();
        for &id in ids {
            joined.extend_from_slice(
                self.token_bytes(id)
                    .ok_or_else(|| eyre!("token {id} out of range"))?,
            );
        }

        Ok(String::from_utf8_lossy(&joined).into_owned())
    }

    /// Id of a special token by its text, such as `<|transcribe|>`
    pub fn special(&self, name: &str) -> Option<Token> {
        self.specials.get(name).copied()
    }

    /// Rank (token id) of an ordinary token's bytes
    pub fn rank(&self, bytes: &[u8]) -> Option<Token> {
        self.ranks.get(bytes).copied()
    }

    /// `encoding.encode_single_token(text)` for an ordinary token
    pub fn single_token(&self, text: &str) -> Option<Token> {
        self.ranks
            .get(text.as_bytes())
            .copied()
            .or_else(|| self.special(text))
    }

    pub fn eot(&self) -> Token {
        self.specials["<|endoftext|>"]
    }

    pub fn sot(&self) -> Token {
        self.specials["<|startoftranscript|>"]
    }

    pub fn transcribe(&self) -> Token {
        self.specials["<|transcribe|>"]
    }

    pub fn no_timestamps(&self) -> Token {
        self.specials["<|notimestamps|>"]
    }

    pub fn timestamp_begin(&self) -> Token {
        self.specials["<|0.00|>"]
    }

    /// `Tokenizer.all_language_tokens`. Python builds it from a set, so its order is
    /// hash order, but only membership is ever used
    pub fn all_language_tokens(&self) -> Vec<Token> {
        LANGUAGES[..self.num_languages]
            .iter()
            .map(|l| self.specials[&format!("<|{l}|>")])
            .collect()
    }

    /// `Tokenizer.to_language_token(language)`
    pub fn language_token(&self, language: &str) -> Result<Token> {
        self.special(&format!("<|{language}|>"))
            .ok_or_else(|| eyre!("Language {language} not found in tokenizer."))
    }
}

#[cfg(test)]
mod tests {
    use super::Tokenizer;

    #[test]
    fn special_token_layout() {
        let tok = Tokenizer::multilingual().unwrap();
        assert_eq!(tok.n_vocab(), 51866);
        assert_eq!(tok.eot(), 50257);
        assert_eq!(tok.sot(), 50258);
        assert_eq!(tok.language_token("en").unwrap(), 50259);
        assert_eq!(tok.transcribe(), 50360);
        assert_eq!(tok.timestamp_begin(), 50365);
        assert_eq!(tok.special("<|30.00|>"), Some(51865));
    }
}
