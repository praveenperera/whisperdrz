//! The decoder forward of a speculative verify step ([`Decoder::verify`])
//!
//! A verify step packs rows of 1 to [`MAX_QUERIES`] tokens: each row's last written token
//! then its drafts, at consecutive positions from the row's cache length. The step runs
//! the one-token step's sequence over the packed tokens, every linear on [`Gemm::Decode`]
//! at the packed row count, the attentions on [`VerifyAttn`]. Padding tokens are rows of
//! their own on the trash slot at length 0. The step leaves the per-slot cache lengths
//! alone: only the host learns how many drafts each row kept, and the next one-token
//! step's positions say so ([`Workspace::sync_lens`]).
//!
//! The LM head runs only over the tokens whose logits the host reads, gathered and padded
//! to their bucket, so a step whose drafts are all forced runs the head over one row per
//! slot, as a one-token step does

use eyre::{Result, ensure};

use super::super::decoder_attn::verify::{MAX_QUERIES, Ragged, VerifyAttn};
use super::super::decoder_attn::{Batch, KvCaches, Rows};
use super::super::device::{Arg, Buf, Gpu, blocks};
use super::super::{Blas, Event, HostBuf};
use super::{D_MODEL, Decoder, Gemm, N_LAYER, N_TEXT_CTX, N_VOCAB, Workspace, i32s, linear_step};
use crate::transcriber::VerifyStep;

/// Logits rows per download chunk of a verify step, as for the one-token step
pub const VERIFY_CHUNK_ROWS: usize = 4;

/// A verify step's inputs on the device, at offsets into [`VerifyState`]'s one upload
#[derive(Clone, Copy, Debug)]
pub struct Staged {
    tokens: u64,
    positions: u64,
    rows: Ragged,
    /// `[head_rows]` packed token per LM-head row, -1 for padding
    head_idx: u64,
    head_rows: usize,
    /// LM-head rows the host reads, the first `need` of them
    pub need: usize,
}

/// What verify steps keep between steps: the input array, the attention's split
/// workspace, and the page-locked logits with an event per [`VERIFY_CHUNK_ROWS`] rows
pub struct VerifyState {
    input: Buf,
    input_words: usize,
    max_tokens: usize,
    pub attn: VerifyAttn,
    pub host: HostBuf,
    pub events: Vec<Event>,
}

impl VerifyState {
    /// Buffers for steps of up to `max_tokens` packed tokens
    pub fn new(gpu: &Gpu, max_tokens: usize) -> Result<Self> {
        // tokens and positions, then per row (at most one per token) its slot, start and
        // length plus the closing start, then the LM head's token per row
        let input_words = 6 * max_tokens + 1;
        Ok(Self {
            input: gpu.zeros(input_words * 4)?,
            input_words,
            max_tokens,
            attn: VerifyAttn::new(gpu, max_tokens)?,
            host: gpu.alloc_host(max_tokens * N_VOCAB * 2)?,
            events: (0..max_tokens.div_ceil(VERIFY_CHUNK_ROWS))
                .map(|_| gpu.event())
                .collect::<Result<_>>()?,
        })
    }

    /// Device bytes held
    pub fn bytes(&self) -> usize {
        self.input.bytes() + self.attn.bytes()
    }

    /// Writes `step`'s inputs to the device in one upload, padding rows on `trash`, and
    /// returns where they sit. Synchronizes, as [`Workspace::stage`] does
    pub fn stage(&self, gpu: &Gpu, step: &VerifyStep, trash: usize) -> Result<Staged> {
        let tokens = step.bucket();
        let real = step.real_tokens();
        ensure!(
            step.rows >= 1
                && step.slots.len() == step.rows
                && step.starts.len() == step.rows + 1
                && step.starts[0] == 0
                && step.starts.windows(2).all(|w| w[1] > w[0])
                && real <= tokens
                && step.positions.len() == tokens,
            "verify step layout: {} rows, starts {:?}, {} tokens, {} positions",
            step.rows,
            step.starts,
            tokens,
            step.positions.len()
        );
        ensure!(
            tokens <= self.max_tokens && step.head_rows <= self.max_tokens,
            "verify step of {tokens} tokens and {} LM-head rows; at most {}",
            step.head_rows,
            self.max_tokens
        );
        ensure!(
            step.need.len() <= step.head_rows
                && step.need.windows(2).all(|w| w[1] > w[0])
                && step.need.last().is_none_or(|&j| j < real),
            "verify step needs {:?} of {real} tokens over {} LM-head rows",
            step.need,
            step.head_rows
        );
        ensure!(
            step.tokens.iter().all(|&t| (t as usize) < N_VOCAB)
                && step
                    .positions
                    .iter()
                    .all(|&p| (0..N_TEXT_CTX as i32).contains(&p)),
            "verify token past the vocabulary or position outside 0..{N_TEXT_CTX}"
        );
        let max_q = step.max_row_len();
        ensure!(
            max_q <= MAX_QUERIES,
            "verify row of {max_q} tokens; at most {MAX_QUERIES}"
        );
        for r in 0..step.rows {
            let at = step.starts[r];
            let first = step.positions[at];
            ensure!(
                (at..step.starts[r + 1]).all(|j| step.positions[j] == first + (j - at) as i32),
                "verify row {r} positions {:?} are not consecutive",
                &step.positions[at..step.starts[r + 1]]
            );
        }

        // the real rows, then one row per padding token on the trash slot at length 0
        let pad = tokens - real;
        let rows = step.rows + pad;
        let slots = i32s(
            step.slots
                .iter()
                .copied()
                .chain(std::iter::repeat_n(trash, pad)),
            "slot",
        )?;
        let starts = i32s(
            step.starts.iter().copied().chain(real + 1..=tokens),
            "start",
        )?;
        let lens: Vec<i32> = (0..step.rows)
            .map(|r| step.positions[step.starts[r]])
            .chain(std::iter::repeat_n(0, pad))
            .collect();
        let head = i32s(step.need.iter().copied(), "token")?
            .into_iter()
            .chain(std::iter::repeat_n(-1, step.head_rows - step.need.len()));

        let mut words: Vec<i32> = Vec::with_capacity(self.input_words);
        words.extend(i32s(step.tokens.iter().map(|&t| t as usize), "token")?);
        words.extend_from_slice(&step.positions);
        let at_slots = words.len();
        words.extend_from_slice(&slots);
        let at_starts = words.len();
        words.extend_from_slice(&starts);
        let at_lens = words.len();
        words.extend_from_slice(&lens);
        let at_head = words.len();
        words.extend(head);
        ensure!(
            words.len() <= self.input_words,
            "verify inputs of {} words",
            words.len()
        );
        gpu.upload_to(self.input.ptr(), bytemuck::cast_slice(&words))?;

        let at = |w: usize| self.input.at(w * 4);
        Ok(Staged {
            tokens: at(0),
            positions: at(tokens),
            rows: Ragged {
                slots: at(at_slots),
                starts: at(at_starts),
                lens: at(at_lens),
                rows,
                tokens,
                max_q,
            },
            head_idx: at(at_head),
            head_rows: step.head_rows,
            need: step.need.len(),
        })
    }
}

impl Decoder {
    /// The forward of a staged verify step: the embedding, 32 blocks over the packed
    /// tokens, the final LayerNorm, then the LM head over the needed tokens into the first
    /// `staged.need` rows of [`Workspace::logits`]. The rows' new keys and values go into
    /// their slots' caches; the cache lengths stay as they were
    pub fn verify(
        &self,
        gpu: &Gpu,
        blas: &Blas,
        attn: &VerifyAttn,
        caches: &KvCaches,
        ws: &mut Workspace,
        staged: &Staged,
    ) -> Result<()> {
        self.verify_traced(gpu, blas, attn, caches, ws, staged, |_, _| Ok(()))
    }

    /// [`Decoder::verify`], calling `after_block(i, ws)` once block `i`'s output is in
    /// [`Workspace::x`], for per-layer checks
    #[allow(clippy::too_many_arguments)]
    pub fn verify_traced(
        &self,
        gpu: &Gpu,
        blas: &Blas,
        attn: &VerifyAttn,
        caches: &KvCaches,
        ws: &mut Workspace,
        staged: &Staged,
        mut after_block: impl FnMut(usize, &Workspace) -> Result<()>,
    ) -> Result<()> {
        let n = staged.rows.tokens;
        ensure!(
            n >= 1 && n <= ws.max_tokens && staged.head_rows <= ws.max_tokens,
            "verify step of {n} tokens for a workspace of {}",
            ws.max_tokens
        );
        let (x, h, qkv, o, ao, f) = (
            ws.x.ptr(),
            ws.h.ptr(),
            ws.qkv.ptr(),
            ws.o.ptr(),
            ws.ao.ptr(),
            ws.f.ptr(),
        );
        let small = &ws.small;
        let gemm = Gemm::Decode;

        self.embed_rows(gpu, x, (staged.tokens, staged.positions), n, false)?;
        self.blocks[0].attn_ln.apply(gpu, x, h, n)?;
        // the sub-ops of `Decoder::run_block` in order, attention on the packed rows
        for i in 0..N_LAYER {
            let blk = &self.blocks[i];
            let qkv_dims = (n, D_MODEL, 3 * D_MODEL);
            let ops = (h, blk.qkv_w.ptr(), blk.qkv_b.ptr(), qkv);
            linear_step(gpu, blas, small, gemm, ops, qkv_dims, false)?;
            let packed = |col: usize| Rows {
                ptr: qkv + 2 * col as u64,
                stride: 3 * D_MODEL,
            };
            attn.self_attn(
                gpu,
                caches,
                i,
                staged.rows,
                (packed(0), packed(D_MODEL), packed(2 * D_MODEL)),
                o,
            )?;
            blk.attn_out
                .apply_step(gpu, blas, small, gemm, (o, ao), n, false)?;

            blk.cross_ln.add_apply(gpu, x, ao, h, n)?;
            blk.cross_q
                .apply_step(gpu, blas, small, gemm, (h, qkv), n, false)?;
            attn.cross_attn(gpu, caches, i, staged.rows, Rows::dense(qkv), o)?;
            blk.cross_out
                .apply_step(gpu, blas, small, gemm, (o, ao), n, false)?;

            blk.mlp_ln.add_apply(gpu, x, ao, h, n)?;
            blk.fc1
                .apply_step(gpu, blas, small, gemm, (h, f), n, true)?;
            blk.fc2
                .apply_step(gpu, blas, small, gemm, (f, ao), n, false)?;

            let next = self.blocks.get(i + 1).map_or(&self.ln, |b| &b.attn_ln);
            next.add_apply(gpu, x, ao, h, n)?;
            after_block(i, ws)?;
        }

        // the head's rows of the final LayerNorm, gathered into the free MLP buffer
        let (c8, rows) = (D_MODEL / 8, staged.head_rows);
        gpu.launch(
            "gather_rows_f16x8",
            (blocks(rows * c8, 256), 1, 1),
            (256, 1, 1),
            0,
            &[
                Arg::slice(f, rows * c8),
                Arg::slice(h, n * c8),
                Arg::slice(staged.head_idx, rows),
                c8.into(),
            ],
        )?;
        blas.matmul(
            f,
            self.tok_emb_t.ptr(),
            ws.logits.ptr(),
            (rows, D_MODEL, N_VOCAB),
        )
    }
}

impl Workspace {
    /// Sets the cache length of each staged row's slot to the row's position, and the
    /// trash slot's to 0, for a one-token step after verify steps left the lengths behind
    pub fn sync_lens(&self, gpu: &Gpu, caches: &KvCaches, batch: Batch) -> Result<()> {
        ensure!(
            batch.q_len == 1 && batch.slots == self.slots.ptr(),
            "length sync for a staged one-token step"
        );
        gpu.launch(
            "kv_lens_set",
            (blocks(batch.rows, 64), 1, 1),
            (64, 1, 1),
            0,
            &[
                caches.lens().slice::<i32>(),
                Arg::slice(self.slots.ptr(), batch.rows),
                Arg::slice(self.positions.ptr(), batch.rows),
                caches.trash().into(),
            ],
        )
    }
}
