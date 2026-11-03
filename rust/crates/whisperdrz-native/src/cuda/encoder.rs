//! The Whisper large-v3 audio encoder on the GPU: FP16 mel `[B, 128, 3000]` to FP16
//! features `[B, 1500, 1280]`, the reference's `AudioEncoder.forward`
//!
//! Stem: `gelu(conv1(mel))` (kernel 3, padding 1), `gelu(conv2(.))` (kernel 3, stride 2,
//! padding 1), then the positional embedding. Each conv runs as an im2col gather and one
//! cuBLASLt GEMM against the weight viewed as `[1280, c_in * 3]`; conv1's GELU is fused
//! into conv2's gather and conv2's GELU into the positional add. Then 32 pre-LN blocks,
//! `x += attn(attn_ln(x))` and `x += mlp(mlp_ln(x))`, and `ln_post`. Per block: one QKV
//! GEMM on the concatenated query, key and value weights (the key's missing bias is a
//! zero bias, an exact no-op), the flash attention kernel straight on the packed QKV rows,
//! the out projection, a fused residual add and `mlp_ln`, fc1, GELU, fc2, and a fused
//! residual add with the next block's `attn_ln` (or `ln_post` after the last block)
//!
//! Every value is rounded to FP16 wherever torch rounds it, so the only differences from
//! the reference are summation order inside GEMMs and attention

use eyre::{Result, ensure};

use super::device::{Arg, Buf, Gpu, blocks};
use super::ops::{self, HeadView};
use super::{Blas, Tensor, Weights};

pub const N_MELS: usize = 128;
/// Mel frames per 30 s chunk
pub const N_FRAMES: usize = 3000;
/// Audio positions after the stride-2 conv
pub const N_CTX: usize = 1500;
pub const D_MODEL: usize = 1280;
pub const N_HEAD: usize = 20;
pub const N_LAYER: usize = 32;
pub const D_MLP: usize = 4 * D_MODEL;

/// SDPA's scale for the reference's `(d_head ** -0.25) ** 2`, exactly 1/8 in FP32
const ATTN_SCALE: f32 = 0.125;

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
}

/// One residual attention block's weights
struct Block {
    attn_ln: Norm,
    /// `[3 * 1280, 1280]`: query, key and value weights stacked
    qkv_w: Buf,
    /// `[3 * 1280]`: query bias, zeros for the bias-free key, value bias
    qkv_b: Buf,
    out_w: Tensor,
    out_b: Tensor,
    mlp_ln: Norm,
    fc1_w: Tensor,
    fc1_b: Tensor,
    fc2_w: Tensor,
    fc2_b: Tensor,
}

impl Block {
    fn take(gpu: &Gpu, weights: &mut Weights, i: usize) -> Result<Self> {
        let p = format!("encoder.blocks.{i}");
        let square = [D_MODEL, D_MODEL];
        let bytes = D_MODEL * D_MODEL * 2;
        let qkv_w = gpu.alloc(3 * bytes)?;
        let qkv_b = gpu.zeros(3 * D_MODEL * 2)?;
        for (j, part) in ["query", "key", "value"].into_iter().enumerate() {
            let w = weights.take(&format!("{p}.attn.{part}.weight"), &square)?;
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
            out_w: weights.take(&format!("{p}.attn.out.weight"), &square)?,
            out_b: weights.take(&format!("{p}.attn.out.bias"), &[D_MODEL])?,
            mlp_ln: Norm::take(weights, &format!("{p}.mlp_ln"))?,
            fc1_w: weights.take(&format!("{p}.mlp.0.weight"), &[D_MLP, D_MODEL])?,
            fc1_b: weights.take(&format!("{p}.mlp.0.bias"), &[D_MLP])?,
            fc2_w: weights.take(&format!("{p}.mlp.2.weight"), &[D_MODEL, D_MLP])?,
            fc2_b: weights.take(&format!("{p}.mlp.2.bias"), &[D_MODEL])?,
        })
    }
}

/// Activation buffers for one batch size, reused across blocks and calls
pub struct Workspace {
    batch: usize,
    /// The residual stream `[B, 1500, 1280]`
    x: Buf,
    /// LayerNorm output, the input of the next GEMM
    h: Buf,
    /// Packed QKV rows `[B, 1500, 3840]`; the stem's im2col matrices before that
    qkv: Buf,
    /// Attention output; conv2's output in the stem
    o: Buf,
    /// Out projection and fc2 output, the residual branch
    ao: Buf,
    /// MLP hidden `[B, 1500, 5120]`; conv1's output in the stem
    f: Buf,
}

impl Workspace {
    pub fn new(gpu: &Gpu, batch: usize) -> Result<Self> {
        ensure!(batch >= 1, "encoder workspace for an empty batch");
        let rows = batch * N_CTX;
        Ok(Self {
            batch,
            x: gpu.alloc(rows * D_MODEL * 2)?,
            h: gpu.alloc(rows * D_MODEL * 2)?,
            qkv: gpu.alloc(rows * 3 * D_MODEL * 2)?,
            o: gpu.alloc(rows * D_MODEL * 2)?,
            ao: gpu.alloc(rows * D_MODEL * 2)?,
            f: gpu.alloc(rows * D_MLP * 2)?,
        })
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    /// The residual stream `[B, 1500, 1280]`: the stem's output after [`Encoder::stem`],
    /// block `i`'s after [`Encoder::block`]
    pub fn x(&self) -> &Buf {
        &self.x
    }
}

/// The encoder's weights on the GPU
pub struct Encoder {
    conv1_w: Tensor,
    conv1_b: Tensor,
    conv2_w: Tensor,
    conv2_b: Tensor,
    pos: Tensor,
    blocks: Vec<Block>,
    ln_post: Norm,
}

impl Encoder {
    /// Takes the `encoder.*` tensors out of `weights`, repacking each block's query, key
    /// and value weights into one QKV matrix
    pub fn new(gpu: &Gpu, weights: &mut Weights) -> Result<Self> {
        let blocks = (0..N_LAYER)
            .map(|i| Block::take(gpu, weights, i))
            .collect::<Result<Vec<_>>>()?;
        let this = Self {
            conv1_w: weights.take("encoder.conv1.weight", &[D_MODEL, N_MELS, 3])?,
            conv1_b: weights.take("encoder.conv1.bias", &[D_MODEL])?,
            conv2_w: weights.take("encoder.conv2.weight", &[D_MODEL, D_MODEL, 3])?,
            conv2_b: weights.take("encoder.conv2.bias", &[D_MODEL])?,
            pos: weights.take("encoder.positional_embedding", &[N_CTX, D_MODEL])?,
            blocks,
            ln_post: Norm::take(weights, "encoder.ln_post")?,
        };
        // the repacking copies read the originals, which drop as this returns
        gpu.synchronize()?;
        Ok(this)
    }

    /// The whole encoder: `mel` `[B, 128, 3000]` to `out` `[B, 1500, 1280]`, both FP16,
    /// with `B = ws.batch()`. `out` must not overlap `mel` or the workspace
    pub fn forward(
        &self,
        gpu: &Gpu,
        blas: &Blas,
        mel: u64,
        out: u64,
        ws: &mut Workspace,
    ) -> Result<()> {
        self.stem(gpu, blas, mel, ws)?;
        let rows = ws.batch * N_CTX;
        let first = &self.blocks[0].attn_ln;
        ops::layernorm(
            gpu,
            ws.x.ptr(),
            first.w.ptr(),
            first.b.ptr(),
            ws.h.ptr(),
            (rows, D_MODEL),
        )?;
        for i in 0..N_LAYER {
            let (next, h_out) = match self.blocks.get(i + 1) {
                Some(b) => (&b.attn_ln, ws.h.ptr()),
                None => (&self.ln_post, out),
            };
            self.run_block(gpu, blas, i, ws, Some((next, h_out)))?;
        }

        Ok(())
    }

    /// The stem alone: `mel` `[B, 128, 3000]` to the residual stream [`Workspace::x`]
    pub fn stem(&self, gpu: &Gpu, blas: &Blas, mel: u64, ws: &mut Workspace) -> Result<()> {
        let b = ws.batch;
        // conv1: [B * 3000, 128 * 3] im2col of the channel-major mel
        im2col(
            gpu,
            ws.qkv.ptr(),
            mel,
            b,
            (N_FRAMES, N_FRAMES, N_MELS, 1),
            (N_MELS * N_FRAMES, N_FRAMES, 1),
            false,
        )?;
        blas.linear_bias(
            ws.qkv.ptr(),
            self.conv1_w.ptr(),
            self.conv1_b.ptr(),
            ws.f.ptr(),
            (b * N_FRAMES, N_MELS * 3, D_MODEL),
        )?;
        // conv2: [B * 1500, 1280 * 3] im2col of gelu(conv1), which is time-major
        im2col(
            gpu,
            ws.qkv.ptr(),
            ws.f.ptr(),
            b,
            (N_FRAMES, N_CTX, D_MODEL, 2),
            (N_FRAMES * D_MODEL, 1, D_MODEL),
            true,
        )?;
        blas.linear_bias(
            ws.qkv.ptr(),
            self.conv2_w.ptr(),
            self.conv2_b.ptr(),
            ws.o.ptr(),
            (b * N_CTX, D_MODEL * 3, D_MODEL),
        )?;

        let n8 = b * N_CTX * D_MODEL / 8;
        gpu.launch(
            "gelu_pos_f16x8",
            (blocks(n8, 256), 1, 1),
            (256, 1, 1),
            0,
            &[
                Arg::slice(ws.x.ptr(), n8),
                Arg::slice(ws.o.ptr(), n8),
                Arg::slice(self.pos.ptr(), N_CTX * D_MODEL / 8),
            ],
        )
    }

    /// Block `i` alone on the residual stream: `ws.x = block_i(ws.x)`, through the same
    /// kernels [`Encoder::forward`] runs, for per-layer checks
    pub fn block(&self, gpu: &Gpu, blas: &Blas, i: usize, ws: &mut Workspace) -> Result<()> {
        ensure!(i < N_LAYER, "encoder block {i}");
        let ln = &self.blocks[i].attn_ln;
        let rows = ws.batch * N_CTX;
        ops::layernorm(
            gpu,
            ws.x.ptr(),
            ln.w.ptr(),
            ln.b.ptr(),
            ws.h.ptr(),
            (rows, D_MODEL),
        )?;
        self.run_block(gpu, blas, i, ws, None)
    }

    /// `ln_post` alone: `out = ln_post(ws.x)`
    pub fn ln_post(&self, gpu: &Gpu, ws: &Workspace, out: u64) -> Result<()> {
        let ln = &self.ln_post;
        ops::layernorm(
            gpu,
            ws.x.ptr(),
            ln.w.ptr(),
            ln.b.ptr(),
            out,
            (ws.batch * N_CTX, D_MODEL),
        )
    }

    /// Block `i` with `ws.h = attn_ln_i(ws.x)` already computed. The final residual add
    /// is fused with `next`'s LayerNorm into `next.1` when given, else a plain add
    fn run_block(
        &self,
        gpu: &Gpu,
        blas: &Blas,
        i: usize,
        ws: &mut Workspace,
        next: Option<(&Norm, u64)>,
    ) -> Result<()> {
        let blk = &self.blocks[i];
        let (b, rows) = (ws.batch, ws.batch * N_CTX);
        blas.linear_bias(
            ws.h.ptr(),
            blk.qkv_w.ptr(),
            blk.qkv_b.ptr(),
            ws.qkv.ptr(),
            (rows, D_MODEL, 3 * D_MODEL),
        )?;

        let packed = (b, N_CTX, 3 * D_MODEL);
        ops::attention(
            gpu,
            HeadView::rows(ws.o.ptr(), (b, N_CTX, D_MODEL), 0),
            HeadView::rows(ws.qkv.ptr(), packed, 0),
            HeadView::rows(ws.qkv.ptr(), packed, D_MODEL),
            HeadView::rows(ws.qkv.ptr(), packed, 2 * D_MODEL),
            (b, N_HEAD, N_CTX, N_CTX),
            ATTN_SCALE,
        )?;
        blas.linear_bias(
            ws.o.ptr(),
            blk.out_w.ptr(),
            blk.out_b.ptr(),
            ws.ao.ptr(),
            (rows, D_MODEL, D_MODEL),
        )?;
        let ln = &blk.mlp_ln;
        ops::add_layernorm(
            gpu,
            ws.x.ptr(),
            ws.ao.ptr(),
            ln.w.ptr(),
            ln.b.ptr(),
            ws.h.ptr(),
            (rows, D_MODEL),
        )?;

        blas.linear_bias(
            ws.h.ptr(),
            blk.fc1_w.ptr(),
            blk.fc1_b.ptr(),
            ws.f.ptr(),
            (rows, D_MODEL, D_MLP),
        )?;
        ops::gelu_inplace(gpu, ws.f.ptr(), rows * D_MLP)?;
        blas.linear_bias(
            ws.f.ptr(),
            blk.fc2_w.ptr(),
            blk.fc2_b.ptr(),
            ws.ao.ptr(),
            (rows, D_MLP, D_MODEL),
        )?;
        match next {
            Some((ln, h_out)) => ops::add_layernorm(
                gpu,
                ws.x.ptr(),
                ws.ao.ptr(),
                ln.w.ptr(),
                ln.b.ptr(),
                h_out,
                (rows, D_MODEL),
            ),
            None => ops::add_inplace(gpu, ws.x.ptr(), ws.ao.ptr(), rows * D_MODEL),
        }
    }
}

/// Launches `im2col_k3_f16` for a kernel-3, padding-1 conv: `out` `[batch * t_out, c_in *
/// 3]` from input element `(b, c, t)` at `x[b * bs + c * cs + t * ts]`
fn im2col(
    gpu: &Gpu,
    out: u64,
    x: u64,
    batch: usize,
    (t_in, t_out, c_in, stride): (usize, usize, usize, usize),
    (bs, cs, ts): (usize, usize, usize),
    gelu: bool,
) -> Result<()> {
    let n = batch * t_out * c_in * 3;
    let x_len = (batch - 1) * bs + (c_in - 1) * cs + (t_in - 1) * ts + 1;
    gpu.launch(
        "im2col_k3_f16",
        (blocks(n, 256), 1, 1),
        (256, 1, 1),
        0,
        &[
            Arg::slice(out, n),
            Arg::slice(x, x_len),
            t_in.into(),
            t_out.into(),
            c_in.into(),
            stride.into(),
            bs.into(),
            cs.into(),
            ts.into(),
            Arg::U32(gelu as u32),
        ],
    )
}
