//! ECAPA2 speaker-embedding glue in FP32: everything between the cuBLAS GEMMs that carry
//! the convolutions, the DFT and the final linear layer
//!
//! The 2D trunk keeps activations as `[rows][channels][cols]` (frequency-major, time
//! innermost) with a one-element zero border on rows and columns when `pad` is 1, so a
//! 3x3 convolution is three strided-batched GEMMs reading shifted views of the buffer.
//! Kernels that write such a buffer cover its whole extent and write zeros on the border,
//! which keeps every conv input's padding zero. The 1D head uses `[channels][frames]` with
//! `pad` 0. Batch norm arrives as one `[3][channels]` slice: running mean, then
//! `weight / sqrt(var + eps)`, then bias, applied as torch does, `(x - mean) * k + b`

use cuda_device::cooperative_groups::ops::{Max, Sum};
use cuda_device::cooperative_groups::{ThreadGroup, block_reduce, this_thread_block};
use cuda_device::{BlockRows, DisjointSlice, SharedArray, kernel, launch_contract, thread};

/// Threads per block of the block-per-row kernels
const BLOCK: u32 = 256;
const WARPS: usize = (BLOCK / 32) as usize;
/// Hidden width of every squeeze-excitation MLP in ECAPA2
pub const SE_HIDDEN: usize = 128;

/// Index of padded position `i` under reflect padding (no edge repeat) of `len` samples;
/// the host guarantees the pad is shorter than `len`
#[inline(always)]
fn reflect(i: i32, len: i32) -> u32 {
    let i = if i < 0 { -i } else { i };
    (if i >= len { 2 * (len - 1) - i } else { i }) as u32
}

/// Batch norm in inference, as torch computes it: `(x - mean) * k + b` with
/// `k = weight / sqrt(var + eps)`; `bn` is `[mean; k; b]` over `channels`
#[inline(always)]
fn batch_norm(v: f32, bn: &[f32], c: usize, channels: usize) -> f32 {
    (v - bn[c]) * bn[channels + c] + bn[2 * channels + c]
}

/// torch's ReLU: negative to zero, NaN kept so it fails loudly downstream
#[inline(always)]
fn relu(v: f32) -> f32 {
    if v < 0.0 { 0.0 } else { v }
}

/// STFT frames for `torch.stft(center=True, pad_mode="reflect")` with a window shorter
/// than the transform: `out[t][n] = x[reflect(t * hop + n - n_fft / 2)] * w[n - offset]`,
/// zero outside the window, `[frames][n_fft]`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn ecapa_frames(
    mut out: DisjointSlice<f32>,
    x: &[f32],
    window: &[f32],
    n_fft: u32,
    hop: u32,
    offset: u32,
) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let Some(mut out) = out.element_thread32(t) else { return };
    let (frame, n) = (i / n_fft, i % n_fft);
    let v = if n >= offset && n - offset < window.len() as u32 {
        let pos = (frame * hop + n) as i32 - (n_fft / 2) as i32;
        x[reflect(pos, x.len() as i32) as usize] * window[(n - offset) as usize]
    } else {
        0.0
    };
    out.write(v);
}

/// `log(|X|^2 + 1e-6)` from the DFT GEMM's `[frames][2 * bins]` output (real parts, then
/// imaginary parts), written frequency-major as `[bins][frames]`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn ecapa_log_power(mut out: DisjointSlice<f32>, y: &[f32], frames: u32, bins: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let Some(mut out) = out.element_thread32(t) else { return };
    let (k, f) = (i / frames, i % frames);
    let base = (f * 2 * bins + k) as usize;
    let (re, im) = (y[base], y[base + bins as usize]);
    out.write(libm::logf(re * re + im * im + 1e-6));
}

/// Each row of `x` (`[rows - 2 pad][cols]`) minus its mean, into the padded trunk input
/// `out` (`[rows][cols + 2 pad]`, one block per output row, zero border)
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
pub fn ecapa_center_rows(
    mut out: DisjointSlice<f32, BlockRows>,
    x: &[f32],
    rows: u32,
    cols: u32,
    pad: u32,
) {
    static mut SMEM: SharedArray<f32, WARPS> = SharedArray::UNINIT;
    let Some(mut row) = out.block_row32(lc) else { return };
    let r = row.row();
    if pad != 0 && (r == 0 || r >= rows - 1) {
        for mut y in row.thread_columns() {
            y.write(0.0);
        }
        return;
    }
    let base = ((r - pad) * cols) as usize;
    if base + cols as usize > x.len() {
        return;
    }

    let block = this_thread_block();
    let mut sum = 0.0f32;
    let mut c = thread::threadIdx_x();
    while c < cols {
        sum += x[base + c as usize];
        c += BLOCK;
    }
    // SAFETY: every thread of the block reaches this call (the returns above are uniform
    // per block) and only it uses SMEM
    let sum = unsafe { block_reduce::<f32, Sum, WARPS>(&block, sum, &raw mut SMEM) };
    // torch's MeanOps: the sum times 1 / n
    let mean = sum * (1.0 / cols as f32);
    for mut y in row.thread_columns() {
        let w = y.column();
        let v = if w < pad || w >= cols + pad { 0.0 } else { x[base + (w - pad) as usize] - mean };
        y.write(v);
    }
}

/// True for a border element of a `[rows][channels][cols]` buffer with a `pad` border
#[inline(always)]
fn on_border(h: u32, w: u32, rows: u32, cols: u32, pad: u32) -> bool {
    pad != 0 && (h == 0 || h == rows - 1 || w == 0 || w == cols - 1)
}

/// The conv epilogue in place over a whole `[rows][channels][cols]` buffer: add `bias`,
/// ReLU when `relu` is 1, then batch norm `bn` (each optional: an empty slice skips it),
/// with zeros on the border
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn ecapa_conv_epilogue(
    mut y: DisjointSlice<f32>,
    bias: &[f32],
    bn: &[f32],
    rows: u32,
    channels: u32,
    cols: u32,
    pad: u32,
    relu_on: u32,
) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let Some(mut y) = y.element_thread32(t) else { return };
    let w = i % cols;
    let c = (i / cols) % channels;
    let h = i / (cols * channels);
    if on_border(h, w, rows, cols, pad) {
        y.write(0.0);
        return;
    }

    let c = c as usize;
    let mut v = y.read();
    if !bias.is_empty() {
        v += bias[c];
    }
    if relu_on != 0 {
        v = relu(v);
    }
    if !bn.is_empty() {
        v = batch_norm(v, bn, c, channels as usize);
    }
    y.write(v);
}

/// Per-channel mean over the interior of a `[rows][channels][cols]` buffer (one block
/// per channel, `out` is `[channels][1]`): squeeze-excitation's average pool
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
pub fn ecapa_channel_mean(
    mut out: DisjointSlice<f32, BlockRows>,
    x: &[f32],
    rows: u32,
    channels: u32,
    cols: u32,
    pad: u32,
) {
    static mut SMEM: SharedArray<f32, WARPS> = SharedArray::UNINIT;
    let Some(mut row) = out.block_row32(lc) else { return };
    let c = row.row();
    let (h_in, w_in) = (rows - 2 * pad, cols - 2 * pad);
    let n = h_in * w_in;
    if x.len() < (rows * channels * cols) as usize {
        return;
    }

    let block = this_thread_block();
    let mut sum = 0.0f32;
    let mut j = thread::threadIdx_x();
    while j < n {
        let (h, w) = (j / w_in + pad, j % w_in + pad);
        sum += x[((h * channels + c) * cols + w) as usize];
        j += BLOCK;
    }
    // SAFETY: every thread of the block reaches this call and only it uses SMEM
    let sum = unsafe { block_reduce::<f32, Sum, WARPS>(&block, sum, &raw mut SMEM) };
    for mut y in row.thread_columns() {
        y.write(sum * (1.0 / n as f32));
    }
}

/// Squeeze-excitation's MLP for one input: `sigmoid(W2 bn(relu(W1 m + b1)) + b2)` with
/// `W1` `[SE_HIDDEN][channels]` and `W2` `[channels][SE_HIDDEN]`; one block, `out` is one
/// row of `channels`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
pub fn ecapa_se_fc(
    mut out: DisjointSlice<f32, BlockRows>,
    mean: &[f32],
    w1: &[f32],
    b1: &[f32],
    bn: &[f32],
    w2: &[f32],
    b2: &[f32],
) {
    static mut HIDDEN: SharedArray<f32, SE_HIDDEN> = SharedArray::UNINIT;
    let Some(mut row) = out.block_row32(lc) else { return };
    let channels = row.width() as usize;
    if mean.len() != channels
        || w1.len() != SE_HIDDEN * channels
        || w2.len() != channels * SE_HIDDEN
        || b1.len() != SE_HIDDEN
        || b2.len() != channels
        || bn.len() != 3 * SE_HIDDEN
    {
        return;
    }

    // SAFETY: HIDDEN is this kernel's own shared array
    let hidden = unsafe { SharedArray::as_raw_mut_ptr(&raw mut HIDDEN) };
    let mut r = thread::threadIdx_x() as usize;
    while r < SE_HIDDEN {
        let mut z = b1[r];
        for c in 0..channels {
            z += w1[r * channels + c] * mean[c];
        }
        let z = batch_norm(relu(z), bn, r, SE_HIDDEN);
        // SAFETY: each thread writes only its own rows r, and the barrier below orders
        // these writes before any read
        unsafe { *hidden.add(r) = z };
        r += BLOCK as usize;
    }
    thread::sync_threads();

    for mut y in row.thread_columns() {
        let c = y.column() as usize;
        let mut s = b2[c];
        for r in 0..SE_HIDDEN {
            // SAFETY: every row was written before the barrier and none is written after
            s += w2[c * SE_HIDDEN + r] * unsafe { *hidden.add(r) };
        }
        y.write(1.0 / (1.0 + libm::expf(-s)));
    }
}

/// `out = x * s[channel] + r` over a whole `[rows][channels][cols]` buffer with zeros on
/// the border: squeeze-excitation's rescale plus the residual (`r` empty for none)
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn ecapa_scale_residual(
    mut out: DisjointSlice<f32>,
    x: &[f32],
    s: &[f32],
    r: &[f32],
    rows: u32,
    channels: u32,
    cols: u32,
    pad: u32,
) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let Some(mut out) = out.element_thread32(t) else { return };
    let w = i % cols;
    let c = (i / cols) % channels;
    let h = i / (cols * channels);
    if on_border(h, w, rows, cols, pad) {
        out.write(0.0);
        return;
    }

    let i = i as usize;
    let v = x[i] * s[c as usize];
    out.write(if r.is_empty() { v } else { v + r[i] });
}

/// Res2Net group input: `a + b` (`b` empty for the first group) over `[channels][frames]`,
/// reflect-padded by `pad` frames on both sides into `[channels][frames + 2 pad]`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn ecapa_reflect_add(mut out: DisjointSlice<f32>, a: &[f32], b: &[f32], frames: u32, pad: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let Some(mut out) = out.element_thread32(t) else { return };
    let width = frames + 2 * pad;
    let (c, j) = (i / width, i % width);
    let src = (c * frames + reflect(j as i32 - pad as i32, frames as i32)) as usize;
    let v = if b.is_empty() { a[src] } else { a[src] + b[src] };
    out.write(v);
}

/// Attentive pooling's context input `[x; mean; std]`, `[3 channels][frames]`, one block
/// per output row: rows of `x`, then each channel's mean over time, then
/// `sqrt(var + 1e-5)` with the unbiased variance, broadcast over time
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
pub fn ecapa_stats_cat(mut out: DisjointSlice<f32, BlockRows>, x: &[f32], channels: u32) {
    static mut SMEM: SharedArray<f32, WARPS> = SharedArray::UNINIT;
    let Some(mut row) = out.block_row32(lc) else { return };
    let frames = row.width();
    let (part, c) = (row.row() / channels, row.row() % channels);
    let base = (c * frames) as usize;
    if base + frames as usize > x.len() {
        return;
    }
    if part == 0 {
        for mut y in row.thread_columns() {
            y.write(x[base + y.column() as usize]);
        }
        return;
    }

    let block = this_thread_block();
    let mut sum = 0.0f32;
    let mut t = thread::threadIdx_x();
    while t < frames {
        sum += x[base + t as usize];
        t += BLOCK;
    }
    // SAFETY: every thread of the block reaches this call (the returns above are uniform
    // per block) and only these calls use SMEM, with a barrier between them
    let sum = unsafe { block_reduce::<f32, Sum, WARPS>(&block, sum, &raw mut SMEM) };
    let value = if part == 1 {
        sum * (1.0 / frames as f32)
    } else {
        let mean = sum / frames as f32;
        block.sync();
        let mut sq = 0.0f32;
        let mut t = thread::threadIdx_x();
        while t < frames {
            let d = x[base + t as usize] - mean;
            sq += d * d;
            t += BLOCK;
        }
        // SAFETY: as above, after the barrier that ends the first reduction's reads
        let sq = unsafe { block_reduce::<f32, Sum, WARPS>(&block, sq, &raw mut SMEM) };
        libm::sqrtf(sq / (frames - 1) as f32 + 1e-5)
    };
    for mut y in row.thread_columns() {
        y.write(value);
    }
}

/// Attentive statistics pooling for one channel per block: softmax of `logits` over
/// time as weights, `mu = sum(w x)`, `sigma = sqrt(|sum(w x x) - mu^2| + 1e-5)`, then the
/// pooled batch norm `bn` (`[3][2 channels]`, mu channels first); `out` is
/// `[channels][2]` as (mu, sigma) pairs
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
pub fn ecapa_attentive_pool(
    mut out: DisjointSlice<f32, BlockRows>,
    logits: &[f32],
    x: &[f32],
    bn: &[f32],
    frames: u32,
) {
    static mut SMEM: SharedArray<f32, WARPS> = SharedArray::UNINIT;
    let Some(mut row) = out.block_row32(lc) else { return };
    let c = row.row();
    let channels = bn.len() / 6;
    let base = (c * frames) as usize;
    if base + frames as usize > x.len() || base + frames as usize > logits.len() {
        return;
    }

    let block = this_thread_block();
    let mut max = f32::NEG_INFINITY;
    let mut t = thread::threadIdx_x();
    while t < frames {
        max = max.max(logits[base + t as usize]);
        t += BLOCK;
    }
    // SAFETY: every thread of the block reaches each reduction (the returns above are
    // uniform per block), and a barrier separates each reuse of SMEM
    let max = unsafe { block_reduce::<f32, Max, WARPS>(&block, max, &raw mut SMEM) };
    block.sync();

    let mut sum = 0.0f32;
    let mut t = thread::threadIdx_x();
    while t < frames {
        sum += libm::expf(logits[base + t as usize] - max);
        t += BLOCK;
    }
    // SAFETY: as above
    let sum = unsafe { block_reduce::<f32, Sum, WARPS>(&block, sum, &raw mut SMEM) };
    block.sync();

    let (mut mu, mut sq) = (0.0f32, 0.0f32);
    let mut t = thread::threadIdx_x();
    while t < frames {
        let i = base + t as usize;
        let xs = libm::expf(logits[i] - max) / sum * x[i];
        mu += xs;
        sq += xs * x[i];
        t += BLOCK;
    }
    // SAFETY: as above
    let mu = unsafe { block_reduce::<f32, Sum, WARPS>(&block, mu, &raw mut SMEM) };
    block.sync();
    // SAFETY: as above
    let sq = unsafe { block_reduce::<f32, Sum, WARPS>(&block, sq, &raw mut SMEM) };
    let sigma = libm::sqrtf(libm::fabsf(sq - mu * mu) + 1e-5);

    let c = c as usize;
    for mut y in row.thread_columns() {
        let v = if y.column() == 0 {
            batch_norm(mu, bn, c, 2 * channels)
        } else {
            batch_norm(sigma, bn, channels + c, 2 * channels)
        };
        y.write(v);
    }
}
