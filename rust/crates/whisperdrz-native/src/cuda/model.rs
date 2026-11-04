//! The transcriber's GPU side: one device holding the Whisper encoder and decoder, the
//! per-slot KV and encoder-output caches, and the ECAPA2 speaker embedder, behind
//! [`crate::transcriber::Model`]
//!
//! Slots follow the reference's `--pad-fix` layout: `slots` real slots plus a trash slot
//! that padding rows of a bucketed decode step write to. The caches hold each slot's
//! encoder output `[1500, 1280]` (the reference's `self.xa_cache`), which the DTW forward
//! reads per row and the absorbed cross-attention of large decode buckets reads per
//! layer. Prefill runs on the encoder's contiguous output for the batch it just encoded,
//! as the reference passes `xa` straight from `model.encoder` to `prefill_flash`, so the
//! cross K/V GEMMs see torch's row count

use std::path::Path;
use std::sync::Arc;

use eyre::{Result, ensure};
use half::f16;

use super::decoder::{self, Decoder, MAX_ROWS, N_LAYER, N_VOCAB, PREFILL_LEN};
use super::decoder_attn::{Batch, DecoderAttn, KvCaches, XA_BYTES};
use super::ecapa2::Precision;
use super::embedder::Embedder;
use super::encoder::{N_FRAMES, N_MELS, Workspace};
use super::{Blas, Buf, Encoder, Gpu, Priority, Weights, nvtx};
use crate::dtw::{DtwBatch, Matrices};
use crate::speaker::EMBED_DIM;
use crate::transcriber::{DecodeStep, Logits, MelBatch, Model};

/// The batch a prefill most recently encoded: its slots, whose outputs sit contiguously
/// in `GpuModel::xa_batch` in this order
#[derive(Debug, Default)]
struct Encoded {
    slots: Vec<usize>,
}

/// Every GPU stage the transcriber calls, on one device and stream
pub struct GpuModel {
    gpu: Gpu,
    blas: Blas,
    encoder: Encoder,
    enc_ws: Workspace,
    decoder: Decoder,
    dec_ws: decoder::Workspace,
    attn: DecoderAttn,
    /// Self- and cross-attention caches and each slot's encoder output
    caches: KvCaches,
    /// `[max_batch, 1500, 1280]` FP16, the encoder output of the last encoded batch
    xa_batch: Buf,
    encoded: Encoded,
    /// `[max_batch, 128, 3000]` FP16 encoder input
    mel: Buf,
    /// `i32 [slots + 1]` slot indices for length updates issued from the host
    slot_ids: Buf,
    /// ECAPA2 on its own thread and background stream
    embedder: Embedder,
}

impl GpuModel {
    /// Loads the Whisper checkpoint at `whisper` (without the unused speaker head) and
    /// the exported ECAPA2 weights at `ecapa`, and allocates caches for `slots` real
    /// slots plus the trash slot and an encoder workspace for up to `max_batch` windows
    pub fn new(whisper: &Path, ecapa: &Path, slots: usize, max_batch: usize) -> Result<Self> {
        ensure!(
            (1..=slots).contains(&max_batch),
            "encoder batch {max_batch} for {slots} slots"
        );
        // torch runs ECAPA2's convs in TF32 by default; full FP32 ran on SIMT cores and
        // took over a third of the wall time. FP16 inputs to the trunk convs (FP32 sums)
        // halve the TF32 GEMM time again and keep every layer within 1.5x of torch's
        // TF32 error
        let embedder = Embedder::spawn(0, ecapa, Precision::F16Convs)?;
        // the decoder's stream outranks the embedder's, so speaker embeddings fill the
        // decoder's idle time instead of delaying its steps
        let gpu = Gpu::with_priority(0, Priority::High)?;
        let blas = Blas::new(&gpu)?;
        let mut weights = Weights::load(&gpu, whisper, |name| !name.starts_with("speaker_head"))?;
        let encoder = Encoder::new(&gpu, &mut weights)?;
        let decoder = Decoder::new(&gpu, &mut weights)?;
        ensure!(
            weights.is_empty(),
            "{} checkpoint tensors left unused",
            weights.len()
        );
        Ok(Self {
            enc_ws: Workspace::new(&gpu, max_batch)?,
            // prefills of up to `max_batch` windows into any slots, steps of every bucket
            dec_ws: decoder::Workspace::new(&gpu, MAX_ROWS, PREFILL_LEN, max_batch)?,
            attn: DecoderAttn::new(&gpu)?,
            decoder,
            caches: KvCaches::new(&gpu, N_LAYER, slots)?,
            xa_batch: gpu.alloc(max_batch * XA_BYTES)?,
            encoded: Encoded::default(),
            mel: gpu.alloc(max_batch * N_MELS * N_FRAMES * 2)?,
            slot_ids: gpu.alloc((slots + 1) * 4)?,
            embedder,
            encoder,
            blas,
            gpu,
        })
    }

    /// Writes `slots` as the device slot-index array and returns the batch over them
    fn host_batch(&self, slots: &[usize], q_len: usize) -> Result<Batch> {
        ensure!(
            slots.iter().all(|&s| s <= self.caches.trash()),
            "slots {slots:?} past the trash slot {}",
            self.caches.trash()
        );
        let ids = slots
            .iter()
            .map(|&s| i32::try_from(s))
            .collect::<Result<Vec<_>, _>>()?;
        self.gpu
            .upload_to(self.slot_ids.ptr(), bytemuck::cast_slice(&ids))?;
        Ok(Batch {
            slots: self.slot_ids.ptr(),
            rows: slots.len(),
            q_len,
        })
    }

    /// `reset_flash_kv(slots)`: the slots' self-attention lengths go to 0. Like the
    /// reference, the cached keys and values stay; lengths alone decide what is read
    pub fn reset_kv(&mut self, slots: &[usize]) -> Result<()> {
        if slots.is_empty() {
            return Ok(());
        }

        let batch = self.host_batch(slots, 1)?;
        self.caches.reset_lens(&self.gpu, batch)
    }

    /// `xa = model.encoder(mel)` and `xa_cache[slots] = xa`, keeping `xa` for the
    /// prefill that follows
    pub fn encode(&mut self, mel: &MelBatch, slots: &[usize]) -> Result<()> {
        let rows = slots.len();
        ensure!(
            mel.rows == rows && mel.n_mels == N_MELS && mel.data.len() == rows * N_MELS * N_FRAMES,
            "mel batch of {} rows x {} mels ({} values) for {rows} slots",
            mel.rows,
            mel.n_mels,
            mel.data.len()
        );
        ensure!(
            slots.iter().all(|&s| s < self.caches.slots()),
            "encode into slots {slots:?} of {}",
            self.caches.slots()
        );

        self.enc_ws.set_batch(rows)?;
        self.gpu
            .upload_to(self.mel.ptr(), bytemuck::cast_slice::<f16, u8>(&mel.data))?;
        self.encoder.forward(
            &self.gpu,
            &self.blas,
            self.mel.ptr(),
            self.xa_batch.ptr(),
            &mut self.enc_ws,
        )?;
        self.caches
            .store_xa(&self.gpu, self.xa_batch.ptr(), slots)?;
        self.encoded = Encoded {
            slots: slots.to_vec(),
        };
        Ok(())
    }

    /// `prefill_flash(tokens, xa, ..., slots)`: `tokens` is `[slots.len(), q_len]`
    /// row-major; returns each row's logits at its last position. `xa` is the batch
    /// [`GpuModel::encode`] just produced when its slots match, as in the reference;
    /// otherwise the slots' cached outputs gathered in order
    pub fn prefill(&mut self, tokens: &[u32], slots: &[usize]) -> Result<Logits> {
        ensure!(
            !slots.is_empty() && slots.len() <= self.enc_ws.capacity(),
            "prefill of {} windows, at most {}",
            slots.len(),
            self.enc_ws.capacity()
        );
        if self.encoded.slots != slots {
            for (i, &s) in slots.iter().enumerate() {
                ensure!(s < self.caches.slots(), "prefill slot {s}");
                self.gpu.copy_dtod(
                    self.caches.xa_slot(s)?,
                    self.xa_batch.at(i * XA_BYTES),
                    XA_BYTES,
                )?;
            }
            self.encoded = Encoded {
                slots: slots.to_vec(),
            };
        }

        let batch = self.decoder.prefill(
            &self.gpu,
            &self.blas,
            &self.attn,
            &self.caches,
            &mut self.dec_ws,
            self.xa_batch.ptr(),
            tokens,
            slots,
        )?;
        self.logits(batch.rows, batch.q_len)
    }

    /// `BucketedCUDAGraphs.forward`: one step over the padded batch, returning the real
    /// rows' logits. Padding rows write to the trash slot, which the step resets after
    pub fn decode_step(&mut self, step: &DecodeStep) -> Result<Logits> {
        ensure!(
            step.trash_slot == self.caches.trash(),
            "step pads into slot {}, the trash slot is {}",
            step.trash_slot,
            self.caches.trash()
        );
        ensure!(
            step.rows <= step.slots.len()
                && step.slots[step.rows..]
                    .iter()
                    .all(|&s| s == step.trash_slot)
                && step.slots[..step.rows]
                    .iter()
                    .all(|&s| s < self.caches.slots()),
            "step rows must be real slots followed by trash-slot padding: {:?} ({} real)",
            step.slots,
            step.rows
        );
        let batch = self
            .dec_ws
            .stage(&self.gpu, &step.tokens, &step.positions, &step.slots)?;
        self.decoder.step(
            &self.gpu,
            &self.blas,
            &self.attn,
            &self.caches,
            &mut self.dec_ws,
            batch,
        )?;
        self.logits(step.rows, 1)
    }

    /// The GPU half of `fill_word_timings_batch`, row `i` attending to
    /// `xa_cache[xa_slots[i]]`
    pub fn dtw_prep(&mut self, batch: &DtwBatch, xa_slots: &[usize]) -> Result<Matrices> {
        let xa = xa_slots
            .iter()
            .map(|&s| {
                ensure!(s < self.caches.slots(), "DTW row on slot {s}");
                self.caches.xa_slot(s)
            })
            .collect::<Result<Vec<_>>>()?;
        self.decoder.dtw_prep(&self.gpu, &self.blas, batch, &xa)
    }

    /// The last position's logits of the first `rows` rows of the last prefill or step
    fn logits(&self, rows: usize, q_len: usize) -> Result<Logits> {
        let data = self.dec_ws.last_logits(&self.gpu, rows, q_len)?;
        Logits::new(rows, N_VOCAB, data)
    }

    /// `_embed_audio_segment`: the L2-normalised ECAPA2 embedding of 16 kHz audio
    pub fn embed(&mut self, audio: &[f32]) -> Result<[f32; EMBED_DIM]> {
        self.embedder.embed(audio)
    }
}

impl Model for GpuModel {
    fn reset_kv(&mut self, slots: &[usize]) -> Result<()> {
        let _range = nvtx::Range::new("reset_kv");
        GpuModel::reset_kv(self, slots)
    }

    fn encode(&mut self, mel: &MelBatch, slots: &[usize]) -> Result<()> {
        let _range = nvtx::Range::new("encode");
        GpuModel::encode(self, mel, slots)
    }

    fn prefill(&mut self, tokens: &[u32], slots: &[usize]) -> Result<Logits> {
        let _range = nvtx::Range::new("prefill");
        GpuModel::prefill(self, tokens, slots)
    }

    fn decode_step(&mut self, step: &DecodeStep) -> Result<Logits> {
        let _range = nvtx::Range::new("decode_step");
        GpuModel::decode_step(self, step)
    }

    fn dtw_prep(&mut self, batch: &DtwBatch, xa_slots: &[usize]) -> Result<Matrices> {
        let _range = nvtx::Range::new("dtw_prep");
        GpuModel::dtw_prep(self, batch, xa_slots)
    }

    fn embed(&mut self, audio: &[f32]) -> Result<[f32; EMBED_DIM]> {
        let _range = nvtx::Range::new("embed");
        GpuModel::embed(self, audio)
    }

    fn reset_embeds(&mut self) -> Result<()> {
        self.embedder.reset();
        Ok(())
    }

    fn prefetch_embed(&mut self, audio: &Arc<[f32]>, start: usize, len: usize) -> Result<()> {
        self.embedder.prefetch(audio, start, len)
    }

    fn embed_crop(
        &mut self,
        audio: &Arc<[f32]>,
        start: usize,
        len: usize,
    ) -> Result<[f32; EMBED_DIM]> {
        let _range = nvtx::Range::new("embed");
        self.embedder.embed_crop(audio, start, len)
    }
}
