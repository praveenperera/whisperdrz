//! The Whisper large-v3 text decoder on the GPU: prefill, the decode step and the LM head
//! (`TextDecoder.forward_flash`, `WhisperDRZ.prefill_flash`), and the full-sequence DTW
//! forward with its cross-attention capture and prep (`TextDecoder.forward`,
//! `CrossAttnCapture`, `_dtw_prep_batch`)
//!
//! # Prefill and decode
//!
//! A step runs `rows` rows of `q_len` new tokens, row `i` on KV slot `slots[i]`:
//! `x = f16(token_embedding[tok] + positional_embedding[pos])`, then 32 pre-LN blocks,
//! `x += out(self_attn(attn_ln(x)))`, `x += out(cross_attn(cross_attn_ln(x)))`,
//! `x += fc2(gelu(fc1(mlp_ln(x))))`, the slot lengths advanced once after the last block,
//! then `ln` and the LM head `x @ token_embedding^T` into FP16 logits `[rows * q_len,
//! 51866]`. The self-attention query, key and value weights are packed into one GEMM with
//! a zero key bias, and each residual add is fused with the LayerNorm after it, both exact
//! in real arithmetic; every value is rounded to FP16 where torch rounds it. Decode steps
//! (one token per row) run their linears on the cuBLASLt algorithms measured fastest for
//! the bucket's row count ([`Blas::linear_bias_decode`]), or for the cells
//! [`small_m::route`](super::small_m::route) lists (2 to 16 rows) on the cuda-oxide small-M
//! kernels, fc1's GELU fused there: same FP32 sums and bias epilogue as torch's call, in
//! another order, so outputs can differ from torch's by FP16 rounding. Prefills and the DTW forward keep
//! torch's calls. Prefill first
//! fills every layer's cross keys and values from the encoder output and zeroes the slot
//! lengths, then runs a step at positions `0..q_len`
//!
//! Cross-attention runs one of two ways, chosen per batch shape by
//! [`decoder_attn::uses_absorbed`]. Prefills and small decode buckets attend over the
//! cached cross keys and values like the reference. Large decode buckets attend over the
//! slot's encoder output `xa` instead, with each head's query through its key projection,
//! `q'_h = q_h W_k,h / 8`, and the value projection applied to the softmax sum, `out_h =
//! (softmax(q'_h xa^T) xa) W_v,h^T + b_v,h` ([`DecoderAttn::cross_absorbed`]), exact in
//! real arithmetic since the key has no bias and the weights sum to 1. Its two per-head
//! projections run as head-batched cuBLASLt GEMMs with FP32 sums, `q'` and the attention
//! output rounded to FP16, the value bias in the second GEMM's FP32 epilogue. The
//! reference rounds FP16 keys and values instead, so those outputs differ from torch's by
//! FP16 rounding
//!
//! [`Decoder::step`] is capture safe: it reads tokens, positions and slots from the
//! [`Workspace`]'s device arrays ([`Workspace::stage`] writes them, outside any capture)
//! and enqueues a fixed sequence of launches with no allocation, synchronization or host
//! read, so one CUDA graph per bucket can replay it
//!
//! # DTW
//!
//! [`Decoder::dtw_prep`] is the GPU half of `fill_word_timings_batch`: the causal
//! decoder forward over each row's padded tokens with the timing embedding added, row
//! `i` attending to the encoder output of its slot through the cross keys and values the
//! slot's prefill cached (the same bits the reference's projection gives; see
//! [`Decoder::dtw_scores`]), stopped after the last alignment layer's cross query
//! (nothing later reaches the capture). The capture
//! is `(q * 64^-0.25) . (k * 64^-0.25)` in FP32 per alignment head, from the FP16 query
//! and key projections the reference's hooks see, then the masked softmax, per-column
//! normalization over the real text rows, width-7 median filter and mean over heads

use eyre::{Result, ensure, eyre};
use half::f16;

use super::decoder_attn::{self, Batch, CrossKv, DecoderAttn, KvCaches, Rows};
use super::device::{Arg, Buf, Gpu, blocks};
use super::ops::{self, HeadView};
use super::small_m::{SmallM, SmallStrides};
use super::{Blas, Tensor, Weights};
use crate::dtw::{DtwBatch, Matrices};

pub const D_MODEL: usize = 1280;
pub const N_HEAD: usize = 20;
pub const HEAD_DIM: usize = 64;
pub const N_LAYER: usize = 32;
pub const D_MLP: usize = 4 * D_MODEL;
pub const N_VOCAB: usize = 51866;
/// Text positions, the positional embedding's length
pub const N_TEXT_CTX: usize = 448;
/// Encoder positions each row's cross-attention reads
pub const N_AUDIO_CTX: usize = 1500;
/// First timestamp token (`tokenizer.timestamp_begin`)
pub const TIMESTAMP_BEGIN: u32 = 50365;
/// Timing-embedding rows, 0.00 s to 30.00 s in 0.02 s steps
pub const N_TIMESTAMPS: usize = 1501;

/// Rows of the largest decode bucket and of a prefill batch
pub const MAX_ROWS: usize = 54;
/// Tokens per row in a prefill (the start-of-transcript prompt)
pub const PREFILL_LEN: usize = 7;

/// The `(layer, head)` pairs of large-v3's alignment-head mask
/// (`dtw_align.LARGE_V3_ALIGNMENT_HEADS`) in `head_mask.nonzero()` order
pub const ALIGNMENT_HEADS: [(usize, usize); 10] = [
    (7, 0),
    (10, 17),
    (12, 18),
    (13, 12),
    (16, 1),
    (17, 14),
    (19, 11),
    (21, 4),
    (24, 1),
    (25, 6),
];

/// SDPA's and flash-attn's softmax scale, `64^-0.5`, exactly 1/8 in FP32
const ATTN_SCALE: f32 = 0.125;
/// The capture's per-operand scale `64^-0.25` as torch's FP32 multiply takes it
const CAPTURE_SCALE: f32 = 0.353_553_38;

/// A LayerNorm's weight and bias
struct Norm {
    w: Tensor,
    b: Tensor,
}

impl Norm {
    fn take(weights: &mut Weights, name: &str) -> Result<Self> {
        Ok(Self {
            w: weights.take(&format!("{name}.weight"), &[D_MODEL])?,
            b: weights.take(&format!("{name}.bias"), &[D_MODEL])?,
        })
    }

    /// `out = LN(x)` over `rows` rows
    fn apply(&self, gpu: &Gpu, x: u64, out: u64, rows: usize) -> Result<()> {
        ops::layernorm(gpu, x, self.w.ptr(), self.b.ptr(), out, (rows, D_MODEL))
    }

    /// `x = f16(x + add)` in place, then `out = LN(x)`
    fn add_apply(&self, gpu: &Gpu, x: u64, add: u64, out: u64, rows: usize) -> Result<()> {
        ops::add_layernorm(
            gpu,
            x,
            add,
            self.w.ptr(),
            self.b.ptr(),
            out,
            (rows, D_MODEL),
        )
    }
}

/// A linear layer's weight and optional bias
struct Linear {
    w: Tensor,
    b: Option<Tensor>,
}

impl Linear {
    fn take(
        weights: &mut Weights,
        name: &str,
        (n_out, n_in): (usize, usize),
        bias: bool,
    ) -> Result<Self> {
        Ok(Self {
            w: weights.take(&format!("{name}.weight"), &[n_out, n_in])?,
            b: bias
                .then(|| weights.take(&format!("{name}.bias"), &[n_out]))
                .transpose()?,
        })
    }

    /// `y = x @ w^T (+ b)` over `rows` rows, the call torch makes for this layer
    fn apply(&self, blas: &Blas, x: u64, y: u64, rows: usize) -> Result<()> {
        let dims = (rows, self.w.shape[1], self.w.shape[0]);
        match &self.b {
            Some(b) => blas.linear_bias(x, self.w.ptr(), b.ptr(), y, dims),
            None => blas.linear(x, self.w.ptr(), y, dims),
        }
    }

    /// A step's linear: [`linear_step`] on this layer's weights, then GELU when `gelu`;
    /// bias-free layers always run torch's call
    #[allow(clippy::too_many_arguments)]
    fn apply_step(
        &self,
        gpu: &Gpu,
        blas: &Blas,
        small: &SmallM,
        gemm: Gemm,
        (x, y): (u64, u64),
        rows: usize,
        gelu: bool,
    ) -> Result<()> {
        let (n_out, n_in) = (self.w.shape[0], self.w.shape[1]);
        let dims = (rows, n_in, n_out);
        match &self.b {
            Some(b) => linear_step(
                gpu,
                blas,
                small,
                gemm,
                (x, self.w.ptr(), b.ptr(), y),
                dims,
                gelu,
            ),
            None => {
                blas.linear(x, self.w.ptr(), y, dims)?;
                if gelu {
                    ops::gelu_inplace(gpu, y, rows * n_out)?;
                }
                Ok(())
            }
        }
    }
}

/// `y = x @ w^T + b` over `dims = (rows, n_in, n_out)` on the algorithm `gemm` names, then
/// GELU when `gelu`. Decode steps run the cells [`small_m::route`](super::small_m::route)
/// lists on the small-M kernels, with the GELU fused, and the rest on
/// [`Blas::linear_bias_decode`]
#[allow(clippy::too_many_arguments)]
fn linear_step(
    gpu: &Gpu,
    blas: &Blas,
    small: &SmallM,
    gemm: Gemm,
    (x, w, b, y): (u64, u64, u64, u64),
    dims: (usize, usize, usize),
    gelu: bool,
) -> Result<()> {
    let (rows, n_in, n_out) = dims;
    match gemm {
        Gemm::Torch => blas.linear_bias(x, w, b, y, dims)?,
        Gemm::Decode => match small.route((x, w, y), dims) {
            Some(cfg) => {
                let s = SmallStrides::dense(n_in, n_out);
                return small.linear(gpu, cfg, (x, w, b, y), dims, s, gelu);
            }
            None => blas.linear_bias_decode(x, w, b, y, dims)?,
        },
    }
    if gelu {
        ops::gelu_inplace(gpu, y, rows * n_out)?;
    }
    Ok(())
}

/// Which cuBLASLt algorithm a step's linears run on
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gemm {
    /// torch's heuristic choice, bit for bit the reference's GEMMs
    Torch,
    /// The algorithm measured fastest for the decode buckets ([`Blas::linear_bias_decode`]),
    /// which streams the weights with more blocks at the step's small row counts
    Decode,
}

impl Gemm {
    /// Decode steps (one token per row) run [`Gemm::Decode`]; prefills keep torch's call
    fn for_batch(batch: Batch) -> Self {
        if batch.q_len == 1 {
            Self::Decode
        } else {
            Self::Torch
        }
    }
}

/// One residual attention block's weights
struct Block {
    attn_ln: Norm,
    /// `[3 * 1280, 1280]`: query, key and value weights stacked
    qkv_w: Buf,
    /// `[3 * 1280]`: query bias, zeros for the bias-free key, value bias
    qkv_b: Buf,
    attn_out: Linear,
    cross_ln: Norm,
    cross_q: Linear,
    cross_k: Linear,
    cross_v: Linear,
    cross_out: Linear,
    mlp_ln: Norm,
    fc1: Linear,
    fc2: Linear,
}

impl Block {
    fn take(gpu: &Gpu, weights: &mut Weights, i: usize) -> Result<Self> {
        let p = format!("decoder.blocks.{i}");
        let square = (D_MODEL, D_MODEL);
        let bytes = D_MODEL * D_MODEL * 2;
        let qkv_w = gpu.alloc(3 * bytes)?;
        let qkv_b = gpu.zeros(3 * D_MODEL * 2)?;
        for (j, part) in ["query", "key", "value"].into_iter().enumerate() {
            let w = weights.take(&format!("{p}.attn.{part}.weight"), &[D_MODEL, D_MODEL])?;
            gpu.copy_dtod(w.ptr(), qkv_w.at(j * bytes), bytes)?;
            if part != "key" {
                let b = weights.take(&format!("{p}.attn.{part}.bias"), &[D_MODEL])?;
                gpu.copy_dtod(b.ptr(), qkv_b.at(j * D_MODEL * 2), D_MODEL * 2)?;
            }
        }

        Ok(Self {
            attn_ln: Norm::take(weights, &format!("{p}.attn_ln"))?,
            qkv_w,
            qkv_b,
            attn_out: Linear::take(weights, &format!("{p}.attn.out"), square, true)?,
            cross_ln: Norm::take(weights, &format!("{p}.cross_attn_ln"))?,
            cross_q: Linear::take(weights, &format!("{p}.cross_attn.query"), square, true)?,
            cross_k: Linear::take(weights, &format!("{p}.cross_attn.key"), square, false)?,
            cross_v: Linear::take(weights, &format!("{p}.cross_attn.value"), square, true)?,
            cross_out: Linear::take(weights, &format!("{p}.cross_attn.out"), square, true)?,
            mlp_ln: Norm::take(weights, &format!("{p}.mlp_ln"))?,
            fc1: Linear::take(weights, &format!("{p}.mlp.0"), (D_MLP, D_MODEL), true)?,
            fc2: Linear::take(weights, &format!("{p}.mlp.2"), (D_MODEL, D_MLP), true)?,
        })
    }

    /// The cross-attention key and value weights, for the absorbed attention's two
    /// per-head GEMMs
    fn cross_kv(&self) -> Result<CrossKv> {
        let bv = self
            .cross_v
            .b
            .as_ref()
            .ok_or_else(|| eyre!("cross-attention value without a bias"))?;
        Ok(CrossKv {
            wk: self.cross_k.w.ptr(),
            wv: self.cross_v.w.ptr(),
            bv: bv.ptr(),
        })
    }
}

fn i32s(v: impl IntoIterator<Item = usize>, what: &str) -> Result<Vec<i32>> {
    v.into_iter()
        .map(|x| i32::try_from(x).map_err(|_| eyre!("{what} {x} overflows i32")))
        .collect()
}

/// Activation buffers and the staged step inputs for up to `max_rows * max_q_len` tokens,
/// reused across steps, so a step allocates nothing
pub struct Workspace {
    max_rows: usize,
    max_tokens: usize,
    /// `i32 [max_tokens]` token ids, `[max_tokens]` positions, `[max_rows]` slots
    tokens: Buf,
    positions: Buf,
    slots: Buf,
    /// The residual stream `[tokens, 1280]`
    x: Buf,
    /// LayerNorm output, the input of the next GEMM
    h: Buf,
    /// Packed self-attention QKV rows `[tokens, 3840]`; the cross-attention query after
    qkv: Buf,
    /// Attention output
    o: Buf,
    /// Out projection and fc2 output, the residual branch
    ao: Buf,
    /// MLP hidden `[tokens, 5120]`
    f: Buf,
    /// FP16 logits `[tokens, N_VOCAB]`
    logits: Buf,
    /// Cross K/V for prefills into non-consecutive slots, `[windows * 1500, 1280]`
    cross_scratch: Option<Buf>,
    /// Absorbed cross-attention queries through the key projection `[rows, 20, 1280]`, for
    /// decode steps only
    qk: Buf,
    /// Absorbed cross-attention output before the value projection `[rows, 20, 1280]`
    ok: Buf,
    /// Split-K partials and counters of the small-M linears the decode steps route to
    small: SmallM,
}

impl Workspace {
    /// Buffers for steps of up to `max_rows` rows of `max_q_len` tokens, and prefills of up
    /// to `max_windows` windows into non-consecutive slots (0: consecutive slots only).
    /// [`MAX_ROWS`] by [`PREFILL_LEN`] covers every bucket and prefill
    pub fn new(gpu: &Gpu, max_rows: usize, max_q_len: usize, max_windows: usize) -> Result<Self> {
        ensure!(
            max_rows >= 1 && max_q_len >= 1,
            "decoder workspace for no tokens"
        );
        let n = max_rows * max_q_len;
        let cross = max_windows * N_AUDIO_CTX * D_MODEL * 2;
        Ok(Self {
            max_rows,
            max_tokens: n,
            tokens: gpu.zeros(n * 4)?,
            positions: gpu.zeros(n * 4)?,
            slots: gpu.zeros(max_rows * 4)?,
            x: gpu.alloc(n * D_MODEL * 2)?,
            h: gpu.alloc(n * D_MODEL * 2)?,
            qkv: gpu.alloc(n * 3 * D_MODEL * 2)?,
            o: gpu.alloc(n * D_MODEL * 2)?,
            ao: gpu.alloc(n * D_MODEL * 2)?,
            f: gpu.alloc(n * D_MLP * 2)?,
            logits: gpu.alloc(n * N_VOCAB * 2)?,
            cross_scratch: (max_windows > 0).then(|| gpu.alloc(cross)).transpose()?,
            qk: gpu.alloc(max_rows * N_HEAD * D_MODEL * 2)?,
            ok: gpu.alloc(max_rows * N_HEAD * D_MODEL * 2)?,
            small: SmallM::for_routes(gpu)?,
        })
    }

    /// Writes a step's inputs to the device arrays [`Decoder::step`] reads and returns its
    /// batch: `tokens` and `positions` are `[rows, q_len]` row-major, `slots[i]` is row
    /// `i`'s KV slot. Synchronizes, so call it outside a graph capture
    pub fn stage(
        &mut self,
        gpu: &Gpu,
        tokens: &[u32],
        positions: &[i32],
        slots: &[usize],
    ) -> Result<Batch> {
        let rows = slots.len();
        ensure!(
            rows >= 1 && rows <= self.max_rows,
            "{rows} rows; the workspace holds {}",
            self.max_rows
        );
        ensure!(
            tokens.len().is_multiple_of(rows)
                && tokens.len() >= rows
                && positions.len() == tokens.len()
                && tokens.len() <= self.max_tokens,
            "{} tokens and {} positions for {rows} rows (workspace {} tokens)",
            tokens.len(),
            positions.len(),
            self.max_tokens
        );
        let q_len = tokens.len() / rows;
        ensure!(
            tokens.iter().all(|&t| (t as usize) < N_VOCAB),
            "token id past the {N_VOCAB}-token vocabulary"
        );
        ensure!(
            positions
                .iter()
                .all(|&p| (0..N_TEXT_CTX as i32).contains(&p)),
            "position outside 0..{N_TEXT_CTX}"
        );

        let tok = i32s(tokens.iter().map(|&t| t as usize), "token")?;
        gpu.upload_to(self.tokens.ptr(), bytemuck::cast_slice(&tok))?;
        gpu.upload_to(self.positions.ptr(), bytemuck::cast_slice(positions))?;
        gpu.upload_to(
            self.slots.ptr(),
            bytemuck::cast_slice(&i32s(slots.iter().copied(), "slot")?),
        )?;
        Ok(Batch {
            slots: self.slots.ptr(),
            rows,
            q_len,
        })
    }

    /// The residual stream `[tokens, 1280]`: the embedding after [`Decoder::embed`],
    /// block `i`'s output after [`Decoder::block`]
    pub fn x(&self) -> &Buf {
        &self.x
    }

    /// The final LayerNorm's output after [`Decoder::head`] or [`Decoder::step`]
    pub fn h(&self) -> &Buf {
        &self.h
    }

    /// A sub-layer's output before its residual add, see [`SubOp`]
    pub fn ao(&self) -> &Buf {
        &self.ao
    }

    /// FP16 logits `[tokens, N_VOCAB]` of the last step
    pub fn logits(&self) -> &Buf {
        &self.logits
    }

    /// The logits at the last position of each of the first `rows` rows of a step of
    /// `q_len` tokens per row, `[rows, N_VOCAB]` FP16: what `prefill_flash(...)[:, -1]`
    /// and the decode step return
    pub fn last_logits(&self, gpu: &Gpu, rows: usize, q_len: usize) -> Result<Vec<f16>> {
        ensure!(
            rows * q_len <= self.max_tokens,
            "{rows} rows of {q_len} tokens"
        );
        let mut out = vec![f16::ZERO; rows * N_VOCAB];
        for (r, row) in out.chunks_mut(N_VOCAB).enumerate() {
            let at = (r * q_len + q_len - 1) * N_VOCAB * 2;
            gpu.download_into(&self.logits, at, row)?;
        }
        Ok(out)
    }
}

/// A piece of a decoder block, in the order a block runs them. The LayerNorms read
/// [`Workspace::x`] and write [`Workspace::h`]; the cross and MLP ones first add
/// [`Workspace::ao`] into `x` in place, the residual add before them. The attentions and
/// the MLP read `h` and write `ao`, the output before its residual add
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubOp {
    AttnLn,
    SelfAttn,
    CrossLn,
    CrossAttn,
    MlpLn,
    Mlp,
}

/// The decoder's weights on the GPU
pub struct Decoder {
    tok_emb: Tensor,
    /// `[1280, N_VOCAB]`, the contiguous transposed copy the LM head multiplies by
    /// (`TextDecoder.cache_embedding_weight`)
    tok_emb_t: Buf,
    pos_emb: Tensor,
    /// FP16 `[N_TIMESTAMPS, 1280]` timing embedding of the DTW forward
    timing: Buf,
    blocks: Vec<Block>,
    ln: Norm,
}

impl Decoder {
    /// Takes the `decoder.*` tensors out of `weights`, repacking each block's self-attention
    /// query, key and value weights into one QKV matrix
    pub fn new(gpu: &Gpu, weights: &mut Weights) -> Result<Self> {
        let layers = (0..N_LAYER)
            .map(|i| Block::take(gpu, weights, i))
            .collect::<Result<Vec<_>>>()?;
        let tok_emb = weights.take("decoder.token_embedding.weight", &[N_VOCAB, D_MODEL])?;
        let tok_emb_t = gpu.alloc(N_VOCAB * D_MODEL * 2)?;
        let n = N_VOCAB * D_MODEL;
        gpu.launch(
            "transpose_f16",
            (blocks(n, 256), 1, 1),
            (256, 1, 1),
            0,
            &[
                Arg::slice(tok_emb_t.ptr(), n),
                Arg::slice(tok_emb.ptr(), n),
                N_VOCAB.into(),
                D_MODEL.into(),
            ],
        )?;
        let timing: Vec<u16> = timing_embedding().iter().map(|v| v.to_bits()).collect();
        let this = Self {
            tok_emb,
            tok_emb_t,
            pos_emb: weights.take("decoder.positional_embedding", &[N_TEXT_CTX, D_MODEL])?,
            timing: gpu.upload(timing)?,
            blocks: layers,
            ln: Norm::take(weights, "decoder.ln")?,
        };
        // the repacking copies read the originals, which drop as this returns
        gpu.synchronize()?;
        Ok(this)
    }

    /// The prefill of `prefill_flash`: fills every layer's cross keys and values of the
    /// windows' slots from `xa` (`[slots.len() * 1500, 1280]` FP16, window `i` for
    /// `slots[i]`), zeroes their lengths and runs [`Decoder::step`] on `tokens` (`[rows,
    /// q_len]`) at positions `0..q_len`. The logits land in [`Workspace::logits`]
    #[allow(clippy::too_many_arguments)]
    pub fn prefill(
        &self,
        gpu: &Gpu,
        blas: &Blas,
        attn: &DecoderAttn,
        caches: &KvCaches,
        ws: &mut Workspace,
        xa: u64,
        tokens: &[u32],
        slots: &[usize],
    ) -> Result<Batch> {
        ensure!(
            !slots.is_empty() && tokens.len().is_multiple_of(slots.len()),
            "{} prefill tokens for {} slots",
            tokens.len(),
            slots.len()
        );
        let q_len = tokens.len() / slots.len();
        let positions = i32s((0..slots.len()).flat_map(|_| 0..q_len), "position")?;
        let batch = ws.stage(gpu, tokens, &positions, slots)?;
        self.fill_cross(gpu, blas, caches, ws, xa, slots)?;
        caches.reset_lens(gpu, batch)?;
        self.step(gpu, blas, attn, caches, ws, batch)?;
        Ok(batch)
    }

    /// Every layer's cross keys and values for the prefill windows in `xa`, window `i` into
    /// slot `slots[i]`, as `prefill_flash` computes them, and each window's encoder output
    /// into its slot, which the absorbed cross-attention of large decode buckets reads
    pub fn fill_cross(
        &self,
        gpu: &Gpu,
        blas: &Blas,
        caches: &KvCaches,
        ws: &Workspace,
        xa: u64,
        slots: &[usize],
    ) -> Result<()> {
        ensure!(
            caches.layers() == N_LAYER,
            "caches for {} layers",
            caches.layers()
        );
        let src = caches.store_xa(gpu, xa, slots)?;
        for (i, blk) in self.blocks.iter().enumerate() {
            let bv = blk
                .cross_v
                .b
                .as_ref()
                .ok_or_else(|| eyre!("cross-attention value without a bias"))?
                .ptr();
            caches.fill_cross(
                gpu,
                blas,
                i,
                &src,
                (blk.cross_k.w.ptr(), blk.cross_v.w.ptr(), bv),
                ws.cross_scratch.as_ref(),
            )?;
        }

        Ok(())
    }

    /// One decoder step on the inputs [`Workspace::stage`] wrote: the embedding, 32 blocks,
    /// the slot lengths advanced by `q_len`, the final LayerNorm and the LM head into
    /// [`Workspace::logits`]. Capture safe
    pub fn step(
        &self,
        gpu: &Gpu,
        blas: &Blas,
        attn: &DecoderAttn,
        caches: &KvCaches,
        ws: &mut Workspace,
        batch: Batch,
    ) -> Result<()> {
        let n = Self::tokens(ws, batch)?;
        self.embed(gpu, ws, batch)?;
        self.blocks[0]
            .attn_ln
            .apply(gpu, ws.x.ptr(), ws.h.ptr(), n)?;
        for i in 0..N_LAYER {
            let next = self.blocks.get(i + 1).map_or(&self.ln, |b| &b.attn_ln);
            self.run_block(gpu, blas, attn, caches, i, ws, batch, Some(next))?;
        }

        caches.advance_lens(gpu, batch)?;
        self.lm_head(blas, ws, n)
    }

    fn tokens(ws: &Workspace, batch: Batch) -> Result<usize> {
        let n = batch.rows * batch.q_len;
        ensure!(
            n >= 1 && n <= ws.max_tokens && batch.rows <= ws.max_rows,
            "batch of {} rows x {} tokens for a workspace of {} rows, {} tokens",
            batch.rows,
            batch.q_len,
            ws.max_rows,
            ws.max_tokens
        );
        Ok(n)
    }

    /// The embedding alone: `ws.x = f16(tok_emb[tokens] + pos_emb[positions])`
    pub fn embed(&self, gpu: &Gpu, ws: &Workspace, batch: Batch) -> Result<()> {
        let n = Self::tokens(ws, batch)?;
        self.embed_rows(
            gpu,
            ws.x.ptr(),
            (ws.tokens.ptr(), ws.positions.ptr()),
            n,
            false,
        )
    }

    fn embed_rows(
        &self,
        gpu: &Gpu,
        out: u64,
        (tokens, positions): (u64, u64),
        n: usize,
        timing: bool,
    ) -> Result<()> {
        let (c8, n8) = (D_MODEL / 8, n * D_MODEL / 8);
        let timing = if timing {
            Arg::slice(self.timing.ptr(), N_TIMESTAMPS * c8)
        } else {
            Arg::EMPTY
        };
        gpu.launch(
            "embed_f16x8",
            (blocks(n8, 256), 1, 1),
            (256, 1, 1),
            0,
            &[
                Arg::slice(out, n8),
                Arg::slice(self.tok_emb.ptr(), N_VOCAB * c8),
                Arg::slice(self.pos_emb.ptr(), N_TEXT_CTX * c8),
                timing,
                Arg::slice(tokens, n),
                Arg::slice(positions, n),
                c8.into(),
                TIMESTAMP_BEGIN.into(),
                N_TIMESTAMPS.into(),
            ],
        )
    }

    /// Block `i` alone on the residual stream, `ws.x = block_i(ws.x)`, through the kernels
    /// [`Decoder::step`] runs, for per-layer checks. Appends to the self-attention cache at
    /// the slots' current lengths and doesn't advance them
    #[allow(clippy::too_many_arguments)]
    pub fn block(
        &self,
        gpu: &Gpu,
        blas: &Blas,
        attn: &DecoderAttn,
        caches: &KvCaches,
        i: usize,
        ws: &mut Workspace,
        batch: Batch,
    ) -> Result<()> {
        ensure!(i < N_LAYER, "decoder block {i}");
        let n = Self::tokens(ws, batch)?;
        self.blocks[i]
            .attn_ln
            .apply(gpu, ws.x.ptr(), ws.h.ptr(), n)?;
        self.run_block(gpu, blas, attn, caches, i, ws, batch, None)
    }

    /// The final LayerNorm and LM head alone: `ws.h = ln(ws.x)`, logits `= ws.h @
    /// tok_emb^T` for `tokens` rows
    pub fn head(&self, gpu: &Gpu, blas: &Blas, ws: &mut Workspace, tokens: usize) -> Result<()> {
        ensure!(tokens >= 1 && tokens <= ws.max_tokens, "{tokens} tokens");
        self.ln.apply(gpu, ws.x.ptr(), ws.h.ptr(), tokens)?;
        self.lm_head(blas, ws, tokens)
    }

    /// The LM head alone on the LayerNorm output already in [`Workspace::h`]: logits `=
    /// ws.h @ tok_emb^T` for `tokens` rows
    pub fn lm_head(&self, blas: &Blas, ws: &Workspace, tokens: usize) -> Result<()> {
        ensure!(tokens >= 1 && tokens <= ws.max_tokens, "{tokens} tokens");
        blas.matmul(
            ws.h.ptr(),
            self.tok_emb_t.ptr(),
            ws.logits.ptr(),
            (tokens, D_MODEL, N_VOCAB),
        )
    }

    /// Block `i` with `ws.h = attn_ln_i(ws.x)` already computed. The final residual add is
    /// fused with `next`'s LayerNorm into `ws.h` when given, else a plain add
    #[allow(clippy::too_many_arguments)]
    fn run_block(
        &self,
        gpu: &Gpu,
        blas: &Blas,
        attn: &DecoderAttn,
        caches: &KvCaches,
        i: usize,
        ws: &mut Workspace,
        batch: Batch,
        next: Option<&Norm>,
    ) -> Result<()> {
        use SubOp::{CrossAttn, CrossLn, Mlp, MlpLn, SelfAttn};

        for op in [SelfAttn, CrossLn, CrossAttn, MlpLn, Mlp] {
            self.sub_op(gpu, blas, attn, caches, i, ws, batch, op)?;
        }

        let n = batch.rows * batch.q_len;
        match next {
            Some(ln) => ln.add_apply(gpu, ws.x.ptr(), ws.ao.ptr(), ws.h.ptr(), n),
            None => ops::add_inplace(gpu, ws.x.ptr(), ws.ao.ptr(), n * D_MODEL),
        }
    }

    /// One [`SubOp`] of block `i`, the pieces a block runs in order, for per-sub-op checks
    /// (see [`SubOp`] for the buffers each reads and writes). The self-attention appends to
    /// the cache at the slots' current lengths and doesn't advance them
    #[allow(clippy::too_many_arguments)]
    pub fn sub_op(
        &self,
        gpu: &Gpu,
        blas: &Blas,
        attn: &DecoderAttn,
        caches: &KvCaches,
        i: usize,
        ws: &mut Workspace,
        batch: Batch,
        op: SubOp,
    ) -> Result<()> {
        ensure!(i < N_LAYER, "decoder block {i}");
        let blk = &self.blocks[i];
        let n = Self::tokens(ws, batch)?;
        let (x, h, qkv, o, ao, f) = (
            ws.x.ptr(),
            ws.h.ptr(),
            ws.qkv.ptr(),
            ws.o.ptr(),
            ws.ao.ptr(),
            ws.f.ptr(),
        );

        let gemm = Gemm::for_batch(batch);
        let small = &ws.small;
        match op {
            SubOp::AttnLn => blk.attn_ln.apply(gpu, x, h, n),
            SubOp::SelfAttn => {
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
                    batch,
                    (packed(0), packed(D_MODEL), packed(2 * D_MODEL)),
                    o,
                )?;
                blk.attn_out
                    .apply_step(gpu, blas, small, gemm, (o, ao), n, false)
            }
            SubOp::CrossLn => blk.cross_ln.add_apply(gpu, x, ao, h, n),
            SubOp::CrossAttn => {
                blk.cross_q
                    .apply_step(gpu, blas, small, gemm, (h, qkv), n, false)?;
                if decoder_attn::uses_absorbed(batch.rows, batch.q_len) {
                    let (qk, ok) = (ws.qk.ptr(), ws.ok.ptr());
                    let kv = blk.cross_kv()?;
                    kv.query_keys(blas, qkv, qk, n)?;
                    attn.cross_absorbed(gpu, caches, batch, qk, ok)?;
                    kv.values(blas, ok, o, n)?;
                } else {
                    attn.cross_attn(gpu, caches, i, batch, Rows::dense(qkv), o)?;
                }

                blk.cross_out
                    .apply_step(gpu, blas, small, gemm, (o, ao), n, false)
            }
            SubOp::MlpLn => blk.mlp_ln.add_apply(gpu, x, ao, h, n),
            SubOp::Mlp => {
                blk.fc1
                    .apply_step(gpu, blas, small, gemm, (h, f), n, true)?;
                blk.fc2
                    .apply_step(gpu, blas, small, gemm, (f, ao), n, false)
            }
        }
    }

    /// The GPU half of `fill_word_timings_batch` for the CPU port's `prep` closure: the
    /// decoder forward over `batch.full_tokens`, row `i` attending to the encoder output of
    /// slot `slots[i]` (rows may share one), the alignment-head capture and
    /// `_dtw_prep_batch`. Returns `[rows, t_pad, max_audio_cap]`. Allocates its buffers per
    /// call from the stream pool and synchronizes to download the result
    pub fn dtw_prep(
        &self,
        gpu: &Gpu,
        blas: &Blas,
        caches: &KvCaches,
        batch: &DtwBatch,
        slots: &[usize],
    ) -> Result<Matrices> {
        let scores = self.dtw_scores(gpu, blas, caches, batch, slots)?;
        scores.prep(gpu, batch)
    }

    /// The capture alone: raw FP32 scores `q . k` of every alignment head before the
    /// softmax (`CrossAttnCapture.compute_attention`)
    ///
    /// The cross keys and values are the ones the slots' prefill cached
    /// ([`Decoder::fill_cross`]) rather than a projection of each row's encoder output.
    /// The reference projects them again over the rows stacked, a GEMM with another row
    /// count, but cuBLAS runs both on non-split-K kernels whose FP32 sums take K in the same
    /// order whatever the tile, so each slot's values come out bit for bit the same (checked
    /// by `gpu_dtw`'s `dtw_cached_kv_matches_projection` and the byte-identical benches).
    /// Errors unless every slot's cache holds the prefill of its current encoder output
    pub fn dtw_scores(
        &self,
        gpu: &Gpu,
        blas: &Blas,
        caches: &KvCaches,
        batch: &DtwBatch,
        slots: &[usize],
    ) -> Result<DtwScores> {
        let (rows, t) = (batch.rows(), batch.t_pad);
        ensure!(rows >= 1, "DTW forward over no rows");
        ensure!(
            slots.len() == rows,
            "{} slots for {rows} DTW rows",
            slots.len()
        );
        ensure!(
            (1..=N_TEXT_CTX).contains(&t) && batch.full_tokens.len() == rows * t,
            "DTW batch of {} tokens for {rows} rows of {t} (at most {N_TEXT_CTX})",
            batch.full_tokens.len()
        );
        ensure!(
            batch
                .full_tokens
                .iter()
                .all(|&tok| (tok as usize) < N_VOCAB),
            "DTW token past the vocabulary"
        );
        let last = ALIGNMENT_HEADS.iter().map(|&(l, _)| l).max().unwrap_or(0);
        for &s in slots {
            ensure!(
                s < caches.slots() && caches.cross_from_xa(s, last + 1),
                "DTW row on slot {s}, whose cross keys and values aren't the prefill of its \
                 current encoder output"
            );
        }

        let n = rows * t;
        let tok = i32s(batch.full_tokens.iter().map(|&x| x as usize), "token")?;
        let tokens = gpu.upload(tok)?;
        let positions = gpu.upload(i32s((0..rows).flat_map(|_| 0..t), "position")?)?;
        let (x, h, o, ao) = (
            gpu.alloc(n * D_MODEL * 2)?,
            gpu.alloc(n * D_MODEL * 2)?,
            gpu.alloc(n * D_MODEL * 2)?,
            gpu.alloc(n * D_MODEL * 2)?,
        );
        let qkv = gpu.alloc(n * 3 * D_MODEL * 2)?;
        let f = gpu.alloc(n * D_MLP * 2)?;
        let mut capture = DtwCapture::new(gpu, rows, t)?;
        let runs = slot_runs(slots);

        self.embed_rows(gpu, x.ptr(), (tokens.ptr(), positions.ptr()), n, true)?;
        self.blocks[0].attn_ln.apply(gpu, x.ptr(), h.ptr(), n)?;
        for (li, blk) in self.blocks.iter().enumerate().take(last + 1) {
            blas.linear_bias(
                h.ptr(),
                blk.qkv_w.ptr(),
                blk.qkv_b.ptr(),
                qkv.ptr(),
                (n, D_MODEL, 3 * D_MODEL),
            )?;
            let packed = (rows, t, 3 * D_MODEL);
            ops::attention_causal(
                gpu,
                HeadView::rows(o.ptr(), (rows, t, D_MODEL), 0),
                (
                    HeadView::rows(qkv.ptr(), packed, 0),
                    HeadView::rows(qkv.ptr(), packed, D_MODEL),
                    HeadView::rows(qkv.ptr(), packed, 2 * D_MODEL),
                ),
                (rows, N_HEAD, t),
                ATTN_SCALE,
            )?;
            blk.attn_out.apply(blas, o.ptr(), ao.ptr(), n)?;
            blk.cross_ln.add_apply(gpu, x.ptr(), ao.ptr(), h.ptr(), n)?;

            // the cross query lands in the packed buffer's first `n * 1280` values
            blk.cross_q.apply(blas, h.ptr(), qkv.ptr(), n)?;
            for (s, &(l, head)) in ALIGNMENT_HEADS.iter().enumerate() {
                if l == li {
                    let keys = slots
                        .iter()
                        .map(|&slot| caches.cross_slot(li, slot).map(|(k, _)| k))
                        .collect::<Result<Vec<_>>>()?;
                    capture.gather(gpu, s, qkv.ptr(), &keys, head)?;
                }
            }
            if li == last {
                break;
            }

            for run in &runs {
                let (k, v) = caches.cross_slot(li, slots[run.first])?;
                let cached = |ptr: u64| HeadView {
                    ptr,
                    len: (run.rows - 1) * run.stride * SLOT_ELEMS + SLOT_ELEMS,
                    ld: D_MODEL,
                    batch_stride: run.stride * SLOT_ELEMS,
                };
                let at = (run.first * t * D_MODEL * 2) as u64;
                let tokens = (run.rows, t, D_MODEL);
                ops::attention(
                    gpu,
                    HeadView::rows(o.ptr() + at, tokens, 0),
                    HeadView::rows(qkv.ptr() + at, tokens, 0),
                    cached(k),
                    cached(v),
                    (run.rows, N_HEAD, t, N_AUDIO_CTX),
                    ATTN_SCALE,
                )?;
            }
            blk.cross_out.apply(blas, o.ptr(), ao.ptr(), n)?;
            blk.mlp_ln.add_apply(gpu, x.ptr(), ao.ptr(), h.ptr(), n)?;
            blk.fc1.apply(blas, h.ptr(), f.ptr(), n)?;
            ops::gelu_inplace(gpu, f.ptr(), n * D_MLP)?;
            blk.fc2.apply(blas, f.ptr(), ao.ptr(), n)?;
            self.blocks[li + 1]
                .attn_ln
                .add_apply(gpu, x.ptr(), ao.ptr(), h.ptr(), n)?;
        }

        capture.scores(gpu, blas)
    }
}

/// FP16 values of one slot's cross keys (or values) in one layer, `[1500, 1280]`
const SLOT_ELEMS: usize = N_AUDIO_CTX * D_MODEL;

/// Consecutive DTW rows `first..first + rows` whose slots step by a fixed `stride >= 0`,
/// so one attention launch reads their cached keys and values with that batch stride
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SlotRun {
    first: usize,
    rows: usize,
    stride: usize,
}

/// Splits the rows into the fewest such runs, greedily: a row starts a new run when its
/// slot is below the previous one's or breaks the run's stride
fn slot_runs(slots: &[usize]) -> Vec<SlotRun> {
    let mut runs: Vec<SlotRun> = Vec::new();
    for (i, &s) in slots.iter().enumerate() {
        if let Some(r) = runs.last_mut() {
            let prev = slots[i - 1];
            let fits = s >= prev && (r.rows == 1 || s - prev == r.stride);
            if fits {
                r.stride = s - prev;
                r.rows += 1;
                continue;
            }
        }
        runs.push(SlotRun {
            first: i,
            rows: 1,
            stride: 0,
        });
    }
    runs
}

/// The FP32 query and key heads the capture multiplies, `[heads, rows, t, 64]` and
/// `[heads, rows, 1500, 64]`, filled one alignment head at a time
pub struct DtwCapture {
    rows: usize,
    t_pad: usize,
    q: Buf,
    k: Buf,
    filled: [bool; ALIGNMENT_HEADS.len()],
}

impl DtwCapture {
    /// Buffers for `rows` rows of `t_pad` tokens; [`DtwCapture::scores`] refuses to run
    /// until every alignment head was gathered
    pub fn new(gpu: &Gpu, rows: usize, t_pad: usize) -> Result<Self> {
        let heads = ALIGNMENT_HEADS.len();
        Ok(Self {
            rows,
            t_pad,
            q: gpu.alloc(heads * rows * t_pad * HEAD_DIM * 4)?,
            k: gpu.alloc(heads * rows * N_AUDIO_CTX * HEAD_DIM * 4)?,
            filled: [false; ALIGNMENT_HEADS.len()],
        })
    }

    /// Gathers head `head` of alignment entry `s` from the FP16 cross-attention query
    /// `[rows * t_pad, 1280]` and each row's key projection `keys[b]` (`[1500, 1280]`),
    /// scaled by `64^-0.25` in FP32 as `compute_attention` scales them
    pub fn gather(&mut self, gpu: &Gpu, s: usize, q: u64, keys: &[u64], head: usize) -> Result<()> {
        ensure!(
            s < ALIGNMENT_HEADS.len() && head < N_HEAD,
            "alignment entry {s}, head {head}"
        );
        ensure!(
            keys.len() == self.rows,
            "{} key projections for {} rows",
            keys.len(),
            self.rows
        );
        let q_rows = self.rows * self.t_pad;
        let k_head = N_AUDIO_CTX * HEAD_DIM;
        let k_base = s * self.rows * k_head;
        let parts = std::iter::once((q, self.q.at(s * q_rows * HEAD_DIM * 4), q_rows)).chain(
            keys.iter()
                .enumerate()
                .map(|(b, &k)| (k, self.k.at((k_base + b * k_head) * 4), N_AUDIO_CTX)),
        );
        for (src, dst, n_rows) in parts {
            let n = n_rows * HEAD_DIM;
            gpu.launch(
                "dtw_gather_head_f32",
                (blocks(n, 256), 1, 1),
                (256, 1, 1),
                0,
                &[
                    Arg::slice(dst, n),
                    Arg::slice(src, n_rows * D_MODEL),
                    D_MODEL.into(),
                    (head * HEAD_DIM).into(),
                    CAPTURE_SCALE.into(),
                ],
            )?;
        }

        self.filled[s] = true;
        Ok(())
    }

    /// `scores[s, b] = q[s, b] @ k[s, b]^T`, one FP32 batched GEMM as torch's `Q @
    /// K.transpose(-1, -2)`
    pub fn scores(self, gpu: &Gpu, blas: &Blas) -> Result<DtwScores> {
        ensure!(
            self.filled.iter().all(|&f| f),
            "capture missing alignment heads: {:?}",
            self.filled
        );
        let heads = ALIGNMENT_HEADS.len();
        let buf = gpu.alloc(heads * self.rows * self.t_pad * N_AUDIO_CTX * 4)?;
        blas.sgemm_nt_batched(
            (self.q.ptr(), self.k.ptr(), buf.ptr()),
            (self.t_pad, N_AUDIO_CTX, HEAD_DIM),
            heads * self.rows,
        )?;
        Ok(DtwScores {
            buf,
            heads,
            rows: self.rows,
            t_pad: self.t_pad,
        })
    }
}

/// FP32 cross-attention scores `[heads, rows, t_pad, 1500]` of the alignment heads, the
/// capture's output before `_dtw_prep_batch`; torch's `[rows, heads, t_pad, 1500]` is the
/// same values transposed
pub struct DtwScores {
    buf: Buf,
    heads: usize,
    rows: usize,
    t_pad: usize,
}

impl DtwScores {
    /// Scores from the host in this type's `[heads, rows, t_pad, 1500]` layout, for checks
    /// of the prep on recorded captures
    pub fn upload(
        gpu: &Gpu,
        data: Vec<f32>,
        (heads, rows, t_pad): (usize, usize, usize),
    ) -> Result<Self> {
        ensure!(
            data.len() == heads * rows * t_pad * N_AUDIO_CTX,
            "{} scores for {heads} x {rows} x {t_pad} x {N_AUDIO_CTX}",
            data.len()
        );
        Ok(Self {
            buf: gpu.upload(data)?,
            heads,
            rows,
            t_pad,
        })
    }

    pub fn shape(&self) -> [usize; 4] {
        [self.heads, self.rows, self.t_pad, N_AUDIO_CTX]
    }

    /// The scores as `[heads, rows, t_pad, 1500]`
    pub fn download(&self, gpu: &Gpu) -> Result<Vec<f32>> {
        gpu.download(&self.buf)
    }

    /// `_dtw_prep_batch` in place on the scores: softmax over each row's audio columns
    /// below its cap, standardization over the real text rows, the width-7 median filter
    /// and the mean over heads, cut to `[rows, t_pad, max_audio_cap]`
    pub fn prep(self, gpu: &Gpu, batch: &DtwBatch) -> Result<Matrices> {
        let (heads, rows, t) = (self.heads, self.rows, self.t_pad);
        ensure!(
            batch.rows() == rows
                && batch.t_pad == t
                && batch.audio_caps.len() == rows
                && batch.actual_lens.iter().all(|&n| (1..=t).contains(&n)),
            "DTW batch of {} rows x {} (lens {:?}) for scores of {rows} x {t}",
            batch.rows(),
            batch.t_pad,
            batch.actual_lens
        );
        let width = batch.max_audio_cap();
        ensure!(width <= N_AUDIO_CTX, "audio cap {width}");
        let caps = gpu.upload(i32s(batch.audio_caps.iter().copied(), "audio cap")?)?;
        let lens = gpu.upload(i32s(batch.actual_lens.iter().copied(), "length")?)?;
        let n = heads * rows * t * N_AUDIO_CTX;
        let score_rows = heads * rows * t;

        gpu.launch(
            "dtw_softmax_f32",
            (u32::try_from(score_rows)?, 1, 1),
            (256, 1, 1),
            0,
            &[
                Arg::slice(self.buf.ptr(), n),
                Arg::slice(caps.ptr(), rows),
                score_rows.into(),
                t.into(),
                rows.into(),
                N_AUDIO_CTX.into(),
            ],
        )?;
        let columns = heads * rows * N_AUDIO_CTX;
        gpu.launch(
            "dtw_norm_f32",
            (blocks(columns, 256), 1, 1),
            (256, 1, 1),
            0,
            &[
                Arg::slice(self.buf.ptr(), n),
                Arg::slice(lens.ptr(), rows),
                columns.into(),
                t.into(),
                rows.into(),
                N_AUDIO_CTX.into(),
            ],
        )?;

        let out_n = rows * t * width;
        let out = gpu.alloc(out_n * 4)?;
        // torch's mean kernel multiplies the sum by float(outputs) / inputs, the outputs
        // being the [rows, t, 1500] means before the slice
        let factor = (rows * t * N_AUDIO_CTX) as f32 / n as f32;
        gpu.launch(
            "dtw_median_mean_f32",
            (blocks(out_n, 256), 1, 1),
            (256, 1, 1),
            0,
            &[
                Arg::slice(out.ptr(), out_n),
                Arg::slice(self.buf.ptr(), n),
                heads.into(),
                rows.into(),
                t.into(),
                N_AUDIO_CTX.into(),
                width.into(),
                factor.into(),
            ],
        )?;
        // the scores were standardized in place, so `self` drops with them here
        let data: Vec<f32> = gpu.download(&out)?;
        Ok(Matrices {
            shape: [rows, t, width],
            data,
        })
    }
}

/// `timing_sinusoids(1501, 1280, 30.0)` rounded to FP16, the buffer `TextDecoder` builds
/// in FP32 on the CPU and `model.to(float16)` converts
///
/// torch's steps are FP32: `linspace(0, 30, 1501)`, `exp(-log(10000) / 639 * arange)`,
/// their product, then `sin` and `cos`. The CPU `linspace` kernel runs its scalar path
/// with the multiply-add contracted (`fma(step, i, start)` up to the halfway point,
/// `fma(-step, steps - 1 - i, end)` after it), which this reproduces bit for bit. `exp`,
/// `sin` and `cos` here are correctly rounded (f64, then FP32), where torch's vectorized
/// SLEEF versions are within an ulp. exp's ulp moves an FP32 argument (up to 30) by about
/// 2e-6, so a few FP16 values land on a neighbouring value, several FP16 ulps away where
/// the value is small (58 of 1,921,280 with torch 2.8 on AVX2, at most `30 * 2^-22 + one
/// FP16 ulp` apart). The DTW
/// forward multiplies the table by 0 for every non-timestamp token, and its token rows
/// hold none, so those values never reach an output
pub fn timing_embedding() -> Vec<f16> {
    let half = D_MODEL / 2;
    let steps = N_TIMESTAMPS;
    let (start, end) = (0.0f32, 30.0f32);
    let step = (end - start) / (steps - 1) as f32;
    let times: Vec<f32> = (0..steps)
        .map(|i| {
            if i < steps / 2 {
                step.mul_add(i as f32, start)
            } else {
                (-step).mul_add((steps - 1 - i) as f32, end)
            }
        })
        .collect();
    let inc = -((10000f64).ln() / (half - 1) as f64) as f32;
    let inv: Vec<f32> = (0..half)
        .map(|j| ((inc * j as f32) as f64).exp() as f32)
        .collect();

    let mut out = Vec::with_capacity(steps * D_MODEL);
    for &tv in &times {
        let scaled: Vec<f32> = inv.iter().map(|&v| tv * v).collect();
        out.extend(
            scaled
                .iter()
                .map(|&s| f16::from_f32((s as f64).sin() as f32)),
        );
        out.extend(
            scaled
                .iter()
                .map(|&s| f16::from_f32((s as f64).cos() as f32)),
        );
    }
    out
}
