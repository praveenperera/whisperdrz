//! ECAPA2 speaker-embedding glue in FP32: everything between the cuBLAS GEMMs that carry
//! the convolutions, the DFT and the final linear layer
//!
//! The 2D trunk keeps activations as `[rows][channels][cols]` (frequency-major, time
//! innermost) with a one-element zero border on rows and columns when `pad` is 1. The
//! interior is `width` columns from column `pad`; `cols` is rounded up past
//! `width + 2 pad` to a multiple of 4, so every row starts 16-byte aligned for the GEMMs,
//! and the extra columns count as border. A 3x3 convolution is one GEMM per kernel
//! column stacked into one (`N = 3 co`) over unshifted rows, and
//! [`ecapa_conv3x3_epilogue`] sums the three column-shifted products. Kernels that write
//! such a buffer cover its whole extent and write zeros on the border, which keeps every
//! conv input's padding zero. The 1D head uses `[channels][frames]` with `pad` 0 and
//! `width == cols`. Batch norm arrives as one `[3][channels]` slice: running mean, then
//! `weight / sqrt(var + eps)`, then bias, applied as torch does, `(x - mean) * k + b`

use cuda_device::cooperative_groups::ops::{Max, Sum};
use cuda_device::cooperative_groups::{ThreadGroup, block_reduce, this_thread_block};
use cuda_device::{BlockRows, DisjointSlice, SharedArray, kernel, launch_contract, thread};

use crate::common::f2h;

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
/// `out` (`[rows][at least cols + 2 pad]`, one block per output row, zero border and
/// zeros past the interior)
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
/// around `width` interior columns (the columns past `pad + width` are border too)
#[inline(always)]
fn on_border(h: u32, w: u32, rows: u32, width: u32, pad: u32) -> bool {
    pad != 0 && (h == 0 || h == rows - 1 || w < pad || w >= pad + width)
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
    width: u32,
    relu_on: u32,
) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let Some(mut y) = y.element_thread32(t) else { return };
    let w = i % cols;
    let c = (i / cols) % channels;
    let h = i / (cols * channels);
    if on_border(h, w, rows, width, pad) {
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

/// A 3x3 conv's epilogue over padded output rows `row0..` of a trunk buffer (`out` holds
/// whole rows from `row0`): sums the three kernel columns' products from `y`, then adds
/// `bias`, applies ReLU and batch norm `bn`, with zeros on the border. `y` is the
/// stacked GEMM's `[rows][3 kw][channels][cols]` over interior rows from `yrow0`, each
/// computed at every column of the unshifted input, so output column `w` takes kernel
/// column `kw` from column `w + kw - 1`. Needs `pad` 1
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn ecapa_conv3x3_epilogue(
    mut out: DisjointSlice<f32>,
    y: &[f32],
    bias: &[f32],
    bn: &[f32],
    row0: u32,
    yrow0: u32,
    rows: u32,
    channels: u32,
    cols: u32,
    width: u32,
) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let Some(mut out) = out.element_thread32(t) else { return };
    out.write(conv3x3_output(i, y, bias, bn, (row0, yrow0, rows, channels, cols, width)));
}

/// [`ecapa_conv3x3_epilogue`] writing FP16 (round to nearest even), for an output only
/// the next conv's FP16-input GEMM reads
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn ecapa_conv3x3_epilogue_f16(
    mut out: DisjointSlice<u16>,
    y: &[f32],
    bias: &[f32],
    bn: &[f32],
    row0: u32,
    yrow0: u32,
    rows: u32,
    channels: u32,
    cols: u32,
    width: u32,
) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let Some(mut out) = out.element_thread32(t) else { return };
    let v = conv3x3_output(i, y, bias, bn, (row0, yrow0, rows, channels, cols, width));
    out.write(f2h(v));
}

/// Element `i` of a 3x3 conv epilogue's output (see [`ecapa_conv3x3_epilogue`]): zero
/// on the border, else the three shifted products plus bias, ReLU, batch norm
#[inline(always)]
fn conv3x3_output(
    i: u32,
    y: &[f32],
    bias: &[f32],
    bn: &[f32],
    (row0, yrow0, rows, channels, cols, width): (u32, u32, u32, u32, u32, u32),
) -> f32 {
    let w = i % cols;
    let c = (i / cols) % channels;
    let h = row0 + i / (cols * channels);
    if on_border(h, w, rows, width, 1) {
        return 0.0;
    }

    let plane = (channels * cols) as usize;
    let base = (((h - 1 - yrow0) * 3 * channels + c) * cols + w) as usize;
    // the order the three accumulating GEMMs summed in: kw 0, then 1, then 2
    let v = y[base - 1] + y[base + plane] + y[base + 2 * plane + 1];
    let c = c as usize;
    let v = relu(v + bias[c]);
    batch_norm(v, bn, c, channels as usize)
}

/// Loads in flight per thread in [`ecapa_channel_mean`]
const MEAN_LOADS: usize = 16;

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
    width: u32,
) {
    static mut SMEM: SharedArray<f32, WARPS> = SharedArray::UNINIT;
    let Some(mut row) = out.block_row32(lc) else { return };
    let c = row.row();
    let (h_in, w_in) = (rows - 2 * pad, width);
    let n = h_in * w_in;
    if x.len() < (rows * channels * cols) as usize {
        return;
    }

    let block = this_thread_block();
    let mut sum = 0.0f32;
    let mut j0 = thread::threadIdx_x();
    while j0 < n {
        // MEAN_LOADS of this thread's elements loaded together, then added in the same
        // order as one at a time: the loads overlap and the sum is unchanged
        let mut v = [0.0f32; MEAN_LOADS];
        #[unroll]
        for k in 0..MEAN_LOADS {
            let j = j0 + k as u32 * BLOCK;
            if j < n {
                let (h, w) = (j / w_in + pad, j % w_in + pad);
                v[k] = x[((h * channels + c) * cols + w) as usize];
            }
        }
        #[unroll]
        for k in 0..MEAN_LOADS {
            if j0 + k as u32 * BLOCK < n {
                sum += v[k];
            }
        }
        j0 += MEAN_LOADS as u32 * BLOCK;
    }
    // SAFETY: every thread of the block reaches this call and only it uses SMEM
    let sum = unsafe { block_reduce::<f32, Sum, WARPS>(&block, sum, &raw mut SMEM) };
    for mut y in row.thread_columns() {
        y.write(sum * (1.0 / n as f32));
    }
}

/// Columns of `W1` per shared tile in [`ecapa_se_fc`]
const SE_TC: usize = 64;
/// Hidden units (columns of `W2`) per shared tile in [`ecapa_se_fc`]
const SE_TR: usize = 32;
/// Shared tile words in [`ecapa_se_fc`]: `SE_HIDDEN` rows of `SE_TC + 1` or `BLOCK` rows
/// of `SE_TR + 1`, the extra word per row keeping a warp's walk along its rows off one bank
const SE_TILE: usize = BLOCK as usize * (SE_TR + 1);
/// Loads per thread for one tile of `W1`
const SE_W1_LOADS: usize = SE_HIDDEN * SE_TC / BLOCK as usize;
const _: () = assert!(
    SE_HIDDEN <= BLOCK as usize
        && SE_HIDDEN * (SE_TC + 1) <= SE_TILE
        && (SE_HIDDEN * SE_TC) % BLOCK as usize == 0
        && SE_HIDDEN % SE_TR == 0
);

/// Squeeze-excitation's MLP for one input: `sigmoid(W2 bn(relu(W1 m + b1)) + b2)` with
/// `W1` `[SE_HIDDEN][channels]` and `W2` `[channels][SE_HIDDEN]`; one block, `out` is one
/// row of `channels`
///
/// Each dot product is one thread's, summed in index order as a plain loop would, so the
/// result does not depend on the tiling. The weights reach it through shared tiles the
/// whole block loads with coalesced reads; walking a row straight from global memory
/// would have each lane of a warp read its own cache line
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
    static mut TILE: SharedArray<f32, SE_TILE> = SharedArray::UNINIT;
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

    // SAFETY: HIDDEN and TILE are this kernel's own shared arrays; every write below is
    // ordered before the reads of the same words by a barrier, and every read before the
    // next tile's writes by another
    let hidden = unsafe { SharedArray::as_raw_mut_ptr(&raw mut HIDDEN) };
    let tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE) };
    let tid = thread::threadIdx_x() as usize;
    let block = BLOCK as usize;

    // thread r < SE_HIDDEN sums W1's row r, SE_TC columns per tile
    let mut z = if tid < SE_HIDDEN { b1[tid] } else { 0.0 };
    let mut c0 = 0;
    while c0 < channels {
        // all loads first, so they are in flight together
        let mut v = [0.0f32; SE_W1_LOADS];
        #[unroll]
        for k in 0..SE_W1_LOADS {
            let e = tid + k * block;
            let c = c0 + e % SE_TC;
            if c < channels {
                v[k] = w1[e / SE_TC * channels + c];
            }
        }
        #[unroll]
        for k in 0..SE_W1_LOADS {
            let e = tid + k * block;
            // SAFETY: e / SE_TC < SE_HIDDEN, inside the tile (see SE_TILE)
            unsafe { *tile.add(e / SE_TC * (SE_TC + 1) + e % SE_TC) = v[k] };
        }
        thread::sync_threads();
        if tid < SE_HIDDEN {
            for j in 0..SE_TC.min(channels - c0) {
                // SAFETY: written before the barrier above
                z += unsafe { *tile.add(tid * (SE_TC + 1) + j) } * mean[c0 + j];
            }
        }
        thread::sync_threads();
        c0 += SE_TC;
    }
    if tid < SE_HIDDEN {
        // SAFETY: each thread writes only its own hidden unit
        unsafe { *hidden.add(tid) = batch_norm(relu(z), bn, tid, SE_HIDDEN) };
    }
    thread::sync_threads();

    // thread t sums W2's rows cb + t for each block of BLOCK rows, SE_TR columns per tile
    let mut cols = row.thread_columns();
    let mut cb = 0;
    while cb < channels {
        let c = cb + tid;
        let mut s = if c < channels { b2[c] } else { 0.0 };
        let mut r0 = 0;
        while r0 < SE_HIDDEN {
            // BLOCK rows of SE_TR values, SE_TR loads per thread
            let mut v = [0.0f32; SE_TR];
            #[unroll]
            for k in 0..SE_TR {
                let e = tid + k * block;
                let cc = cb + e / SE_TR;
                if cc < channels {
                    v[k] = w2[cc * SE_HIDDEN + r0 + e % SE_TR];
                }
            }
            #[unroll]
            for k in 0..SE_TR {
                let e = tid + k * block;
                // SAFETY: e / SE_TR < BLOCK, inside the tile
                unsafe { *tile.add(e / SE_TR * (SE_TR + 1) + e % SE_TR) = v[k] };
            }
            thread::sync_threads();
            if c < channels {
                for j in 0..SE_TR {
                    // SAFETY: both written before a barrier above
                    s += unsafe { *tile.add(tid * (SE_TR + 1) + j) * *hidden.add(r0 + j) };
                }
            }
            thread::sync_threads();
            r0 += SE_TR;
        }
        // the walk yields this thread's columns tid, tid + BLOCK, ... in order, the c here
        if let Some(mut y) = cols.next() {
            y.write(1.0 / (1.0 + libm::expf(-s)));
        }
        cb += block;
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
    width: u32,
) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let Some(mut out) = out.element_thread32(t) else { return };
    let w = i % cols;
    let c = (i / cols) % channels;
    let h = i / (cols * channels);
    if on_border(h, w, rows, width, pad) {
        out.write(0.0);
        return;
    }

    let i = i as usize;
    let v = x[i] * s[c as usize];
    out.write(if r.is_empty() { v } else { v + r[i] });
}

/// [`ecapa_scale_residual`] that also writes its output as FP16 (round to nearest even)
/// into `out_h`, the copy the next block's FP16-input conv reads
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn ecapa_scale_residual_f16(
    mut out: DisjointSlice<f32>,
    mut out_h: DisjointSlice<u16>,
    x: &[f32],
    s: &[f32],
    r: &[f32],
    rows: u32,
    channels: u32,
    cols: u32,
    pad: u32,
    width: u32,
) {
    let t = thread::index_1d_u32(lc);
    let th = thread::index_1d_u32(lc);
    let i = t.get();
    let Some(mut out) = out.element_thread32(t) else { return };
    let Some(mut out_h) = out_h.element_thread32(th) else { return };
    let w = i % cols;
    let c = (i / cols) % channels;
    let h = i / (cols * channels);
    let v = if on_border(h, w, rows, width, pad) {
        0.0
    } else {
        let i = i as usize;
        let v = x[i] * s[c as usize];
        if r.is_empty() { v } else { v + r[i] }
    };
    out.write(v);
    out_h.write(f2h(v));
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
