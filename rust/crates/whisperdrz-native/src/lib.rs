//! Native runtime for whisperdrz, a Whisper large-v3 encoder-decoder fine-tuned for speaker
//! tags, word timing and event tags
//!
//! The Python pipeline in `src/whisperdrz` (with the CUDA-graph padding fix) is the
//! reference: every module here reproduces one part of it, and the parity gate compares
//! the final texts and segments byte for byte. Modules that need no GPU (tokenizer,
//! constraint state machine, post-processing, DTW path search, clustering) build without
//! the `cuda` feature and are tested against logged Python inputs and outputs

pub mod text;

#[cfg(feature = "cuda")]
pub mod cuda;
