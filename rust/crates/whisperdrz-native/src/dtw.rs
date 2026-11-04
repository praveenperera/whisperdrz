//! Word timing by DTW over decoder cross-attention, the CPU side of `dtw_align.py`
//!
//! The decoder emits sparse `<|s|>word<|e|>` timestamps; this module fills in the
//! untimed words. For each finalising window it parses the text into speaker lines,
//! words and timed `[tag]` anchors ([`parse_predicted_lines`]), packs the lines into
//! chunks that fit the decoder's positional budget ([`chunk_speaker_lines`]), builds the
//! padded token batch the GPU forward runs on ([`DtwBatch`]), and, given the GPU's
//! normalised attention matrices ([`Matrices`], the output of `_dtw_prep_batch`),
//! bands each chunk's cost matrix around the decoder's own timestamps
//! ([`apply_anchor_bands`]), runs DTW ([`dtw_from_neg_matrix`]), stitches chunks onto
//! the window's timeline and rewrites the text ([`rebuild_with_timings`]).
//!
//! [`fill_word_timings_batch`] is the entry point. It reproduces
//! `dtw_align.fill_word_timings_batch` byte for byte; the decoder forward with
//! cross-attention capture and the softmax/normalise/median prep stay on the GPU side,
//! behind the `prep` closure

use eyre::{Result, ensure};

use crate::text::Tokenizer;

mod batch;
mod chunk;
mod kernel;
mod parse;
mod rebuild;

pub use batch::{
    DtwBatch, DtwOptions, Matrices, MatricesView, PreparedBatch, fill_word_timings_batch,
};
pub use chunk::{CharTokens, ChunkLine, ChunkPrep, TokensChars, chunk_speaker_lines, render_chunk};
pub use kernel::{
    NegMatrix, WordAnchor, WordTime, apply_anchor_bands, dtw_from_neg_matrix, dtw_path,
};
pub use parse::{
    ParsedText, PredWord, Speaker, SpeakerLine, TagAnchor, is_py_space, parse_predicted_lines,
};
pub use rebuild::rebuild_with_timings;

/// Audio positions per second in the encoder output (`audio.TOKENS_PER_SECOND`, 20 ms each)
pub const TOKENS_PER_SECOND: i64 = 50;

/// Frames of slack around each anchor band (`slack_frames=7` in the batched path)
pub const SLACK_FRAMES: i64 = 7;

/// The tokenizer calls DTW needs
///
/// Only single characters and `" "` are encoded, so a char-level table is a complete
/// implementation
pub trait DtwTokenizer {
    /// Token ids for `text`, as `Tokenizer.encode` (tiktoken with all specials allowed)
    fn encode(&self, text: &str) -> Vec<u32>;

    /// `<|endoftext|>`
    fn eot(&self) -> u32;

    /// `<|notimestamps|>`
    fn no_timestamps(&self) -> u32;

    /// `<|startoftranscript|>` plus the language and task tokens
    fn sot_sequence(&self) -> &[u32];
}

/// [`DtwTokenizer`] over the real [`Tokenizer`] with the `en`/`transcribe` start sequence
/// the transcriber's tokenizer uses
pub struct TokenizerDtw<'t> {
    tokenizer: &'t Tokenizer,
    sot_sequence: [u32; 3],
}

impl<'t> TokenizerDtw<'t> {
    pub fn new(tokenizer: &'t Tokenizer) -> Result<Self> {
        // with every single byte ranked, byte-pair encoding can't hit an unranked piece,
        // which is the only way `encode` fails on the one-character inputs DTW sends
        for b in 0..=255u8 {
            ensure!(
                tokenizer.rank(&[b]).is_some(),
                "byte {b} has no rank in the tokenizer"
            );
        }

        Ok(Self {
            tokenizer,
            sot_sequence: [
                tokenizer.sot(),
                tokenizer.language_token("en")?,
                tokenizer.transcribe(),
            ],
        })
    }
}

impl DtwTokenizer for TokenizerDtw<'_> {
    fn encode(&self, text: &str) -> Vec<u32> {
        // DTW encodes single characters and " " only; `new` rules out the failure path
        self.tokenizer
            .encode(text)
            .unwrap_or_else(|e| panic!("encode {text:?} for DTW: {e}"))
    }

    fn eot(&self) -> u32 {
        self.tokenizer.eot()
    }

    fn no_timestamps(&self) -> u32 {
        self.tokenizer.no_timestamps()
    }

    fn sot_sequence(&self) -> &[u32] {
        &self.sot_sequence
    }
}
