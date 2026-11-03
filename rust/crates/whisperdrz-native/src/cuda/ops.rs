//! Transformer building blocks on FP16 device buffers: LayerNorm, GELU, residual adds and
//! bidirectional attention, shared by the encoder and decoder
//!
//! Every function takes plain device addresses and shapes and only enqueues work on the
//! GPU's stream; nothing synchronizes. Buffers are row-major FP16 (`u16` bits). Each
//! kernel computes in FP32 and rounds to FP16 where the torch op it stands for does, and
//! traps (failing the next synchronize) if a shape doesn't fit the lengths passed, so a
//! caller's wrong geometry can't read or write out of bounds. Base addresses must be
//! 16-byte aligned, which every [`Gpu::alloc`] buffer is
//!
//! GEMMs, including the bias adds of `nn.Linear`, go through [`super::Blas`]

use eyre::{Result, ensure};

use super::device::{Arg, Gpu, blocks};

/// `nn.LayerNorm`'s default epsilon, which every Whisper LayerNorm uses
pub const LN_EPS: f32 = 1e-5;

/// Widest LayerNorm row the warp-per-row kernels hold in registers
pub const LN_MAX_COLS: usize = 2048;

/// Head dim of [`attention`]
pub const HEAD_DIM: usize = 64;

/// Threads per block of the 8-values-per-thread kernels
const VEC_BLOCK: u32 = 256;

fn aligned(name: &str, ptr: u64) -> Result<()> {
    ensure!(
        ptr.is_multiple_of(16),
        "{name} at {ptr:#x} is not 16-byte aligned"
    );
    Ok(())
}

fn ln_shape(rows: usize, cols: usize) -> Result<()> {
    ensure!(
        cols.is_multiple_of(8) && (8..=LN_MAX_COLS).contains(&cols),
        "LayerNorm rows of {cols} values; need a multiple of 8 up to {LN_MAX_COLS}"
    );
    ensure!(u32::try_from(rows).is_ok(), "LayerNorm over {rows} rows");
    Ok(())
}

/// `out[r] = (x[r] - mean) * rsqrt(var + LN_EPS) * w + b` for each of `rows` rows of
/// `cols` FP16 values, with FP32 statistics as torch's LayerNorm takes them
///
/// `x` and `out` are `[rows, cols]`, `w` and `b` `[cols]`; `cols` is a multiple of 8 up to
/// [`LN_MAX_COLS`]. `out` must not overlap `x`
pub fn layernorm(
    gpu: &Gpu,
    x: u64,
    w: u64,
    b: u64,
    out: u64,
    (rows, cols): (usize, usize),
) -> Result<()> {
    ln_shape(rows, cols)?;
    for (name, p) in [("x", x), ("w", w), ("b", b), ("out", out)] {
        aligned(name, p)?;
    }

    let (n8, c8) = (rows * cols / 8, cols / 8);
    gpu.launch(
        "layernorm_f16",
        (blocks(rows, 4), 1, 1),
        (128, 1, 1),
        0,
        &[
            Arg::slice(out, n8),
            Arg::slice(x, n8),
            Arg::slice(w, c8),
            Arg::slice(b, c8),
            rows.into(),
            c8.into(),
            LN_EPS.into(),
        ],
    )
}

/// The residual add in front of a pre-LN block, fused with that LayerNorm:
/// `x = f16(x + add)` in place, then `out = LayerNorm(x) * w + b` of the rounded sum, the
/// same values torch's separate add and [`layernorm`] produce
///
/// Shapes as [`layernorm`]; `add` and `out` must not overlap `x` or each other
pub fn add_layernorm(
    gpu: &Gpu,
    x: u64,
    add: u64,
    w: u64,
    b: u64,
    out: u64,
    (rows, cols): (usize, usize),
) -> Result<()> {
    ln_shape(rows, cols)?;
    for (name, p) in [("x", x), ("add", add), ("w", w), ("b", b), ("out", out)] {
        aligned(name, p)?;
    }

    let (n8, c8) = (rows * cols / 8, cols / 8);
    gpu.launch(
        "add_layernorm_f16",
        (blocks(rows, 4), 1, 1),
        (128, 1, 1),
        0,
        &[
            Arg::slice(x, n8),
            Arg::slice(add, n8),
            Arg::slice(out, n8),
            Arg::slice(w, c8),
            Arg::slice(b, c8),
            rows.into(),
            c8.into(),
            LN_EPS.into(),
        ],
    )
}

fn vec_len(op: &str, n: usize) -> Result<usize> {
    ensure!(
        n.is_multiple_of(8),
        "{op} over {n} values; need a multiple of 8"
    );
    Ok(n / 8)
}

/// Exact (erf) GELU in place on `n` FP16 values, `x = f16(gelu(x))` as `nn.GELU()`
/// computes it; `n` is a multiple of 8
pub fn gelu_inplace(gpu: &Gpu, x: u64, n: usize) -> Result<()> {
    let n8 = vec_len("GELU", n)?;
    aligned("x", x)?;
    gpu.launch(
        "gelu_f16x8",
        (blocks(n8, VEC_BLOCK), 1, 1),
        (VEC_BLOCK, 1, 1),
        0,
        &[Arg::slice(x, n8)],
    )
}

/// Residual add in place on `n` FP16 values, `x = f16(x + d)`; `n` is a multiple of 8 and
/// `d` doesn't overlap `x`. Before a LayerNorm, [`add_layernorm`] does this in the same pass
pub fn add_inplace(gpu: &Gpu, x: u64, d: u64, n: usize) -> Result<()> {
    let n8 = vec_len("residual add", n)?;
    aligned("x", x)?;
    aligned("d", d)?;
    gpu.launch(
        "add_f16x8",
        (blocks(n8, VEC_BLOCK), 1, 1),
        (VEC_BLOCK, 1, 1),
        0,
        &[Arg::slice(x, n8), Arg::slice(d, n8)],
    )
}

/// One FP16 operand of [`attention`] as a strided view: element `d` of head `h` of row
/// `i` of batch entry `b` is at element `b * batch_stride + i * ld + h * 64 + d` from
/// `ptr`, and `len` elements from `ptr` are addressable
///
/// For q, k and v packed in one `[batch, rows, 3 * heads * 64]` projection, the three
/// views share `ld` and `batch_stride` and start `heads * 64` elements apart
#[derive(Clone, Copy, Debug)]
pub struct HeadView {
    pub ptr: u64,
    pub len: usize,
    pub ld: usize,
    pub batch_stride: usize,
}

impl HeadView {
    /// A view of a contiguous `[batch, rows, cols]` buffer at `ptr`, or of the columns
    /// from `col` on of each row
    pub fn rows(ptr: u64, (batch, rows, cols): (usize, usize, usize), col: usize) -> Self {
        Self {
            ptr: ptr + 2 * col as u64,
            len: (batch * rows * cols).saturating_sub(col),
            ld: cols,
            batch_stride: rows * cols,
        }
    }

    fn args(&self, name: &str) -> Result<[Arg; 3]> {
        aligned(name, self.ptr)?;
        ensure!(
            self.ld.is_multiple_of(8) && self.batch_stride.is_multiple_of(8),
            "attention {name}: ld {} and batch stride {} must be multiples of 8",
            self.ld,
            self.batch_stride
        );
        let narrow = |v: usize| {
            u32::try_from(v).map_err(|_| eyre::eyre!("attention {name}: {v} overflows u32"))
        };
        Ok([
            Arg::slice(self.ptr, self.len),
            Arg::U32(narrow(self.ld)?),
            Arg::U32(narrow(self.batch_stride)?),
        ])
    }
}

/// Bidirectional multi-head attention, head dim 64:
/// `o[b, i, h] = softmax(scale * q[b, i, h] . k[b, j, h] over j < lk) @ v[b, :, h]`
///
/// The FlashAttention-2 forward pass that torch's SDPA flash backend runs on FP16:
/// FP32-accumulated `q k^T`, online softmax in FP32, P rounded to FP16 for `P @ V` with
/// FP32 accumulation, one division by the row sum and one rounding to FP16 at the end.
/// No mask, any `lq` and `lk >= 1` (the decoder's cross-attention fits, with `lq` its
/// tokens and `lk` the 1500 audio frames). `o` must not overlap q, k or v. Grid:
/// `ceil(lq / 128) x heads x batch` blocks of 256 threads
#[allow(clippy::too_many_arguments)]
pub fn attention(
    gpu: &Gpu,
    o: HeadView,
    q: HeadView,
    k: HeadView,
    v: HeadView,
    (batch, heads, lq, lk): (usize, usize, usize, usize),
    scale: f32,
) -> Result<()> {
    ensure!(lk >= 1, "attention over no keys");
    ensure!(
        batch <= 65535 && heads <= 65535,
        "attention grid of {heads} heads x {batch} batch entries"
    );
    let [o0, o1, o2] = o.args("o")?;
    let [q0, q1, q2] = q.args("q")?;
    let [k0, k1, k2] = k.args("k")?;
    let [v0, v1, v2] = v.args("v")?;
    gpu.launch(
        "flash_attn_f16_d64",
        (blocks(lq, 128), heads as u32, batch as u32),
        (256, 1, 1),
        0,
        &[
            o0,
            o1,
            o2,
            q0,
            q1,
            q2,
            k0,
            k1,
            k2,
            v0,
            v1,
            v2,
            lq.into(),
            lk.into(),
            (scale * std::f32::consts::LOG2_E).into(),
        ],
    )
}
