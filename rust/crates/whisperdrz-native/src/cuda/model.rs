//! The transcriber's GPU side: one device holding the Whisper encoder and decoder, the
//! per-slot KV and encoder-output caches, and the ECAPA2 speaker embedder, behind
//! [`crate::transcriber::Model`]
//!
//! Slots follow the reference's `--pad-fix` layout: `slots` real slots plus a trash slot
//! that padding rows of a bucketed decode step write to. The caches hold each slot's
//! encoder output `[1500, 1280]` (the reference's `self.xa_cache`), which the absorbed
//! cross-attention of large decode buckets reads per layer, and the cross keys and values
//! prefill computed from it, which the DTW forward reads per row. Prefill runs on the encoder's contiguous output for the batch it just encoded,
//! as the reference passes `xa` straight from `model.encoder` to `prefill_flash`, so the
//! cross K/V GEMMs see torch's row count
//!
//! Three streams share the GPU. The decoder's runs at the highest priority. The queued DTW
//! forward ([`GpuModel::dtw_launch`]) runs on a handle of its own one step below, so it
//! fills the SMs the latency-bound decode steps leave instead of stalling them, ordered
//! against the caches by the [`CrossLease`](super::decoder_attn::CrossLease) it reads
//! through. ECAPA2 runs at the lowest priority

use std::cell::Cell;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use eyre::{Result, ensure};
use half::f16;

use super::decoder::{self, Decoder, MAX_ROWS, N_LAYER, N_VOCAB, PREFILL_LEN};
use super::decoder_attn::{Batch, DecoderAttn, KvCaches, XA_BYTES};
use super::ecapa2::Precision;
use super::embedder::Embedder;
use super::encoder::{N_FRAMES, N_MELS, Workspace};
use super::{Blas, Buf, Encoder, Event, Gpu, HostBuf, Priority, UploadMark, Weights, nvtx};
use crate::dtw::{DtwBatch, Matrices, MatricesView};
use crate::speaker::EMBED_DIM;
use crate::transcriber::{
    DecodeStep, DtwMatrices, Logits, MelBatch, Model, ReadyMatrices, StepLogits,
};

/// Logits rows per download chunk of a decode step: the host starts on the first rows
/// while the rest copy, and a chunk copies in less time than its rows take to decode
const STEP_CHUNK_ROWS: usize = 4;

/// The batch a prefill most recently encoded: its slots, whose outputs sit contiguously
/// in `GpuModel::xa_batch` in this order
#[derive(Debug, Default)]
struct Encoded {
    slots: Vec<usize>,
}

/// Every GPU stage the transcriber calls, on one device
pub struct GpuModel {
    gpu: Gpu,
    blas: Blas,
    /// The queued DTW forward's stream, pool and cuBLAS handles
    dtw_gpu: Gpu,
    dtw_blas: Blas,
    /// Marked after the last queued DTW forward's uploads, to free their host copies
    dtw_uploaded: Option<UploadMark>,
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
    /// Page-locked `[MAX_ROWS, N_VOCAB]` FP16 for a decode step's logits, and an event
    /// per [`STEP_CHUNK_ROWS`] rows of it
    step_host: HostBuf,
    step_events: Vec<Event>,
    /// Page-locked buffers of `dtw_gpu` the DTW matrices download into, back here once
    /// their batch is aligned
    dtw_host: Arc<Mutex<Vec<HostBuf>>>,
}

impl Drop for GpuModel {
    fn drop(&mut self) {
        // a queued DTW forward reads the decoder's weights and caches, which drop with the
        // fields after this
        let _ = self.dtw_gpu.synchronize();
    }
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
        // below the decoder, so a queued forward takes the SMs a step leaves rather than
        // delaying it, and above the embedder, since the next step's join waits for it
        let dtw_gpu = Gpu::with_priority(0, Priority::Mid)?;
        let dtw_blas = Blas::new(&dtw_gpu)?;
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
            step_host: gpu.alloc_host(MAX_ROWS * N_VOCAB * 2)?,
            dtw_host: Arc::default(),
            step_events: (0..MAX_ROWS.div_ceil(STEP_CHUNK_ROWS))
                .map(|_| gpu.event())
                .collect::<Result<_>>()?,
            embedder,
            encoder,
            blas,
            gpu,
            dtw_blas,
            dtw_gpu,
            dtw_uploaded: None,
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
        self.queue_step(step)?;
        self.logits(step.rows, 1)
    }

    /// [`Self::decode_step`] without waiting: the step's kernels, then its logits
    /// queued for download into page-locked memory in chunks of [`STEP_CHUNK_ROWS`] rows
    pub fn launch_step(&mut self, step: &DecodeStep) -> Result<InFlight<'_>> {
        self.queue_step(step)?;
        let row_bytes = N_VOCAB * 2;
        let logits = self.dec_ws.logits();
        let chunks = step.rows.div_ceil(STEP_CHUNK_ROWS);
        ensure!(
            chunks <= self.step_events.len(),
            "{} rows for {} download chunks",
            step.rows,
            self.step_events.len()
        );
        // one token a row, so the rows sit contiguously from the start of the logits
        for (c, done) in self.step_events[..chunks].iter().enumerate() {
            let first = c * STEP_CHUNK_ROWS;
            let rows = STEP_CHUNK_ROWS.min(step.rows - first);
            let at = first * row_bytes;
            self.gpu
                .download_async(logits, at, &self.step_host, at, rows * row_bytes, done)?;
        }

        Ok(InFlight {
            host: &self.step_host,
            events: &self.step_events[..chunks],
            rows: step.rows,
            synced: Cell::new(0),
        })
    }

    /// Stages and queues one decode step, leaving its logits on the GPU
    fn queue_step(&mut self, step: &DecodeStep) -> Result<()> {
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
        )
    }

    /// The GPU half of `fill_word_timings_batch`, row `i` attending to
    /// `xa_cache[xa_slots[i]]` through the cross keys and values its prefill cached
    pub fn dtw_prep(&mut self, batch: &DtwBatch, xa_slots: &[usize]) -> Result<Matrices> {
        self.decoder
            .dtw_prep(&self.gpu, &self.blas, &self.caches, batch, xa_slots)
    }

    /// [`GpuModel::dtw_prep`] queued on the DTW stream: the forward and prep, then a copy
    /// of the matrices into page-locked memory, which lands while the host and the
    /// decoder's stream go on. The forward starts after the work queued on the decoder's
    /// stream so far, and the slots' next prefill waits for its reads
    pub fn dtw_launch(
        &mut self,
        batch: &DtwBatch,
        xa_slots: &[usize],
    ) -> Result<Box<dyn DtwMatrices>> {
        // the handle never synchronizes, so the uploads' host copies go once a later
        // event shows the stream past them
        if let Some(uploaded) = &self.dtw_uploaded
            && self.dtw_gpu.release_uploads(uploaded)?
        {
            self.dtw_uploaded = None;
        }

        let (out, shape) = self.decoder.dtw_prep_queued(
            &self.dtw_gpu,
            &self.dtw_blas,
            &self.caches,
            batch,
            xa_slots,
        )?;
        self.dtw_uploaded = Some(self.dtw_gpu.upload_mark()?);
        let bytes = out.bytes();
        if bytes == 0 {
            // no audio column anywhere: nothing to copy
            return Ok(Box::new(ReadyMatrices(Matrices {
                shape,
                data: Vec::new(),
            })));
        }

        let host = self.dtw_host_buf(bytes)?;
        let done = self.dtw_gpu.sleeping_event()?;
        self.dtw_gpu
            .download_async(&out, 0, &host, 0, bytes, &done)?;
        Ok(Box::new(QueuedMatrices {
            host: Some(host),
            done,
            shape,
            pool: self.dtw_host.clone(),
        }))
    }

    /// A free page-locked buffer of at least `bytes`, grown in powers of two so a run
    /// settles on a few
    fn dtw_host_buf(&self, bytes: usize) -> Result<HostBuf> {
        let mut pool = self.dtw_host.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(i) = pool.iter().position(|b| b.bytes() >= bytes) {
            return Ok(pool.swap_remove(i));
        }

        drop(pool);
        self.dtw_gpu
            .alloc_host(bytes.next_power_of_two().max(1 << 20))
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

    fn launch_step<'a>(
        &'a mut self,
        step: &DecodeStep,
        during: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Box<dyn StepLogits + 'a>> {
        let range = nvtx::Range::new("decode_step");
        let logits = GpuModel::launch_step(self, step)?;
        drop(range);
        during()?;
        Ok(Box::new(logits))
    }

    fn dtw_prep(&mut self, batch: &DtwBatch, xa_slots: &[usize]) -> Result<Matrices> {
        let _range = nvtx::Range::new("dtw_prep");
        GpuModel::dtw_prep(self, batch, xa_slots)
    }

    fn dtw_launch(&mut self, batch: &DtwBatch, xa_slots: &[usize]) -> Result<Box<dyn DtwMatrices>> {
        let _range = nvtx::Range::new("dtw_prep");
        GpuModel::dtw_launch(self, batch, xa_slots)
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

/// A decode step's logits on their way into page-locked memory, from
/// [`GpuModel::launch_step`]. It borrows the model, so no later step can overwrite them
pub struct InFlight<'a> {
    host: &'a HostBuf,
    /// One per chunk of [`STEP_CHUNK_ROWS`] rows, recorded after its copy
    events: &'a [Event],
    rows: usize,
    /// Chunks known to have landed, a prefix
    synced: Cell<usize>,
}

impl InFlight<'_> {
    fn row_unchecked(&self, r: usize) -> Result<&[f16]> {
        // SAFETY: row `r`'s chunk has landed (callers check `synced`), and the borrow of
        // the model keeps any later copy from starting while the slice lives
        unsafe { self.host.slice(r * N_VOCAB * 2, N_VOCAB) }
    }
}

impl StepLogits for InFlight<'_> {
    fn rows(&self) -> usize {
        self.rows
    }

    fn row(&self, r: usize) -> Result<&[f16]> {
        ensure!(r < self.rows, "logits row {r} of {}", self.rows);
        let chunk = r / STEP_CHUNK_ROWS;
        while self.synced.get() <= chunk {
            let _range = nvtx::Range::new("logits_wait");
            self.events[self.synced.get()].synchronize()?;
            self.synced.set(self.synced.get() + 1);
        }

        self.row_unchecked(r)
    }

    fn landed(&self) -> Result<usize> {
        while self.synced.get() < self.events.len() && self.events[self.synced.get()].query()? {
            self.synced.set(self.synced.get() + 1);
        }

        Ok((self.synced.get() * STEP_CHUNK_ROWS).min(self.rows))
    }

    fn complete(&self) -> Result<()> {
        if let Some(last) = self.rows.checked_sub(1) {
            self.row(last)?;
        }

        Ok(())
    }
}

/// DTW matrices copying into a page-locked buffer of [`GpuModel`]'s pool
struct QueuedMatrices {
    /// `None` only once dropped
    host: Option<HostBuf>,
    /// Recorded after the copy
    done: Event,
    shape: [usize; 3],
    pool: Arc<Mutex<Vec<HostBuf>>>,
}

impl DtwMatrices for QueuedMatrices {
    fn ready(&self) -> bool {
        false
    }

    fn wait(&mut self) -> Result<MatricesView<'_>> {
        self.done.synchronize_any_thread()?;
        let host = self
            .host
            .as_ref()
            .ok_or_else(|| eyre::eyre!("DTW matrices already returned"))?;
        // SAFETY: the copy into the buffer finished at `done`, and no other copy targets
        // it while this owns it
        let data = unsafe { host.slice::<f32>(0, self.shape.iter().product())? };
        Ok(MatricesView {
            shape: self.shape,
            data,
        })
    }
}

impl Drop for QueuedMatrices {
    fn drop(&mut self) {
        // back for the next batch. Even if nobody waited, the next copy into it is queued
        // behind this one on the same stream, and its reader waits for that copy
        if let Some(host) = self.host.take() {
            self.pool
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(host);
        }
    }
}
