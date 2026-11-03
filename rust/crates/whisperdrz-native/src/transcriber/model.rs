//! The GPU side of the transcriber as a trait: encoder, KV slots, prefill, the decode
//! step, the DTW forward and the speaker embedder

use eyre::{Result, ensure};
use half::f16;

use super::graphs::DecodeStep;
use crate::dtw::{DtwBatch, Matrices};
use crate::speaker::EMBED_DIM;

/// Mel frames per encoder window (`N_FRAMES`, 30 s at 100 frames per second)
pub const N_FRAMES: usize = 3000;

/// Fill value for mel frames past the end of the audio (`F.pad(..., value=-1.5)`)
pub const MEL_PAD: f32 = -1.5;

/// A log-mel spectrogram, `[n_mels, frames]` row-major f32: what `LogMelSpectrogram`
/// returns for one chunk's audio (`tomel(audio.unsqueeze(0)).squeeze(0)`)
#[derive(Debug, Clone, PartialEq)]
pub struct Mel {
    pub n_mels: usize,
    pub frames: usize,
    pub data: Vec<f32>,
}

impl Mel {
    pub fn new(n_mels: usize, frames: usize, data: Vec<f32>) -> Result<Self> {
        ensure!(
            data.len() == n_mels * frames,
            "mel data has {} values, want {n_mels} x {frames}",
            data.len()
        );
        Ok(Self {
            n_mels,
            frames,
            data,
        })
    }

    /// `mel[:, start:start + N_FRAMES]` padded with [`MEL_PAD`] to [`N_FRAMES`]
    /// frames and cast to fp16 (round to nearest even, as torch's `.to(half)`),
    /// appended to `out` row by row
    pub fn window_f16(&self, start: usize, out: &mut Vec<f16>) {
        let start = start.min(self.frames);
        let end = (start + N_FRAMES).min(self.frames);
        let pad = f16::from_f32(MEL_PAD);
        for r in 0..self.n_mels {
            let row = &self.data[r * self.frames..(r + 1) * self.frames];
            out.extend(row[start..end].iter().map(|&v| f16::from_f32(v)));
            out.extend(std::iter::repeat_n(pad, N_FRAMES - (end - start)));
        }
    }
}

/// The encoder input: `[rows, n_mels, N_FRAMES]` fp16, one row per slot
#[derive(Debug, Clone, PartialEq)]
pub struct MelBatch {
    pub rows: usize,
    pub n_mels: usize,
    pub data: Vec<f16>,
}

/// fp16 logits, `[rows, vocab]` row-major
#[derive(Debug, Clone, PartialEq)]
pub struct Logits {
    pub rows: usize,
    pub vocab: usize,
    pub data: Vec<f16>,
}

impl Logits {
    pub fn new(rows: usize, vocab: usize, data: Vec<f16>) -> Result<Self> {
        ensure!(
            data.len() == rows * vocab,
            "logits have {} values, want {rows} x {vocab}",
            data.len()
        );
        Ok(Self { rows, vocab, data })
    }

    pub fn row(&self, r: usize) -> &[f16] {
        &self.data[r * self.vocab..(r + 1) * self.vocab]
    }
}

/// Handle for an asynchronous prefill (a CUDA event recorded on the prefill stream)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrefillEvent(pub u64);

/// Everything the transcriber runs on the GPU
///
/// Slots index the KV cache and the per-slot encoder-output cache (`xa_cache`).
/// There are `n_slots` real slots plus one trash slot (index `n_slots`) that the
/// padded decode step writes its padding rows to (the `--pad-fix` behaviour)
pub trait Model {
    /// `model.reset_flash_kv(self_kv, cross_kv, slots)`: empties the slots' KV
    fn reset_kv(&mut self, slots: &[usize]) -> Result<()>;

    /// `model.encoder(mel)` and `xa_cache[slots] = xa`: encodes one window per slot
    fn encode(&mut self, mel: &MelBatch, slots: &[usize]) -> Result<()>;

    /// `model.prefill_flash(tokens, xa, ..., slots)` with `xa = xa_cache[slots]`:
    /// runs the start-of-transcript `tokens` (`[slots.len(), tokens.len() /
    /// slots.len()]` row-major) through the decoder, filling the slots' KV, and
    /// returns the logits at each row's last position
    fn prefill(&mut self, tokens: &[u32], slots: &[usize]) -> Result<Logits>;

    /// `BucketedCUDAGraphs.forward`: one decoder step over the padded batch,
    /// returning logits for the first `step.rows` rows only. Each real row appends
    /// one position to its slot's KV; padding rows go to the trash slot, whose KV
    /// length the caller expects reset to zero before the step
    fn decode_step(&mut self, step: &DecodeStep) -> Result<Logits>;

    /// The GPU half of `fill_word_timings_batch`: the decoder forward over
    /// `batch.full_tokens` with row `i` attending to `xa_cache[xa_slots[i]]`,
    /// cross-attention capture on the alignment heads, then `_dtw_prep_batch`
    fn dtw_prep(&mut self, batch: &DtwBatch, xa_slots: &[usize]) -> Result<Matrices>;

    /// `_embed_audio_segment`: the L2-normalised ECAPA2 embedding of 16 kHz audio
    fn embed(&mut self, audio: &[f32]) -> Result<[f32; EMBED_DIM]>;

    /// Called before a prefill that runs on the side stream (`n_warm > 0` only)
    fn begin_async_prefill(&mut self) -> Result<()> {
        Ok(())
    }

    /// Records the side-stream event after an async prefill
    fn end_async_prefill(&mut self) -> Result<PrefillEvent> {
        Ok(PrefillEvent(0))
    }

    /// `event.query()`: has the async prefill finished
    fn prefill_done(&mut self, _event: PrefillEvent) -> Result<bool> {
        Ok(true)
    }

    /// `torch.cuda.current_stream().wait_event(event)`
    fn wait_prefill(&mut self, _event: PrefillEvent) -> Result<()> {
        Ok(())
    }
}
