//! The DTW prep on FP32 cross-attention scores: head capture, masked softmax, per-column
//! normalization, and a width-7 median filter fused with the mean over heads
//!
//! `dtw_align.CrossAttnCapture.compute_attention` and `_dtw_prep_batch`, in order. The
//! scores are `w[s, b, t, a]`: alignment head `s`, batch row `b`, text position `t < T`
//! and audio position `a < A` (1500), the layout torch's batched matmul writes before its
//! transpose to `[b, s, t, a]`, which only renames the index here. Each kernel computes in
//! FP32 with the rounding points of the torch ops it stands for; only reduction orders
//! differ

use cuda_device::float::mul_rn_f32;
use cuda_device::thread::{__unroll_config, blockIdx_x, sync_threads, threadIdx_x};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_contract, thread, warp};

use crate::common::h2f;

/// Threads of the block-per-row softmax
const THREADS: usize = 256;
/// Values per thread in the softmax: rows up to 1536
const PER: usize = 6;
/// Median filter width (`medfilt_width`)
const MEDIAN: usize = 7;

/// Traps when the host's geometry doesn't fit the buffers it passed, as a failed slice
/// index would
#[inline(always)]
fn require(ok: bool) {
    if !ok {
        panic!();
    }
}

/// One head of FP16 projection rows widened and scaled, `out[r, d] = f32(src[r * ld + col
/// + d]) * scale` for `d < 64`, as `torch.stack(qs).float() * scale` forms the FP32
/// operands of the capture's matmul (an FP32 product, rounded once)
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn dtw_gather_head_f32(
    mut out: DisjointSlice<f32>,
    src: &[u16],
    ld: u32,
    col: u32,
    scale: f32,
) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let Some(mut out) = out.element_thread32(t) else {
        return;
    };
    let (r, d) = (i / 64, i % 64);
    let x = h2f(src[r as usize * ld as usize + (col + d) as usize]);
    out.write(mul_rn_f32(x, scale));
}

/// Sum over the block, the same value in every thread; `parts` holds one entry per warp
///
/// # Safety
/// Every thread of the block calls it, and nothing else uses `parts` until all have
/// returned from a later barrier
#[inline(always)]
unsafe fn block_reduce(v: f32, parts: *mut f32, max: bool) -> f32 {
    // a counted shift rather than a loop over a literal array, whose iterator stayed on
    // the stack (40 bytes) here even with LLVM 23
    let mut x = v;
    let mut sh = 16;
    while sh >= 1 {
        __unroll_config::<0>();
        let o = warp::shuffle_xor_f32(x, sh);
        x = if max { x.max(o) } else { x + o };
        sh /= 2;
    }
    let tid = threadIdx_x() as usize;
    if tid % 32 == 0 {
        // SAFETY: one slot per warp, THREADS / 32 of them
        unsafe { *parts.add(tid / 32) = x };
    }
    sync_threads();
    // SAFETY: every slot was written before the barrier
    let mut r = unsafe { *parts };
    for k in 1..THREADS / 32 {
        // SAFETY: as above
        let p = unsafe { *parts.add(k) };
        r = if max { r.max(p) } else { r + p };
    }
    r
}

/// Softmax over each score row in place, with audio columns at or past the row's cap set
/// to -inf first (`masked_fill(~audio_mask, -inf)`, then `torch.softmax`): `w = exp(w -
/// max) / sum(exp(w - max))` with the accurate `expf`
///
/// Row `row` (one block each) is `(s * batch + b) * t_len + t` and reads `caps[b]`; rows
/// hold `cols <= 1536` values. A row with no column under its cap comes out NaN, as in
/// torch. Grid `rows` blocks of 256 threads
#[kernel]
#[launch_contract(domain = 1, block = (256, 1, 1))]
pub fn dtw_softmax_f32(
    mut w: DisjointSlice<f32>,
    caps: &[i32],
    rows: u32,
    t_len: u32,
    batch: u32,
    cols: u32,
) {
    static mut MAXES: SharedArray<f32, { THREADS / 32 }> = SharedArray::UNINIT;
    static mut SUMS: SharedArray<f32, { THREADS / 32 }> = SharedArray::UNINIT;
    let row = blockIdx_x();
    require(
        (row < rows)
            & (cols as usize <= THREADS * PER)
            & (t_len >= 1)
            & (batch >= 1)
            & (rows as usize * cols as usize <= w.len())
            & (caps.len() >= batch as usize),
    );
    let b = (row / t_len) % batch;
    // SAFETY: b < batch <= caps.len(), checked above
    let cap = unsafe { *caps.get_unchecked(b as usize) };
    let tid = threadIdx_x() as usize;
    let base = row as usize * cols as usize;
    let wp = w.as_mut_ptr();

    let mut v = [f32::NEG_INFINITY; PER];
    let mut mx = f32::NEG_INFINITY;
    #[unroll]
    for k in 0..PER {
        let c = tid + k * THREADS;
        if c < cols as usize && (c as i32) < cap {
            // SAFETY: base + c < rows * cols <= w.len(); this block alone touches row `row`
            v[k] = unsafe { *wp.add(base + c) };
        }
        mx = mx.max(v[k]);
    }
    // SAFETY: the whole block calls it; MAXES is used for nothing else
    let mx = unsafe { block_reduce(mx, core::ptr::addr_of_mut!(MAXES) as *mut f32, true) };

    let mut s = 0.0f32;
    #[unroll]
    for k in 0..PER {
        if tid + k * THREADS < cols as usize {
            v[k] = libm::expf(v[k] - mx);
            s += v[k];
        }
    }
    // SAFETY: as above, with SUMS
    let s = unsafe { block_reduce(s, core::ptr::addr_of_mut!(SUMS) as *mut f32, false) };

    #[unroll]
    for k in 0..PER {
        let c = tid + k * THREADS;
        if c < cols as usize {
            // SAFETY: inside the checked row, written by this thread only
            unsafe { *wp.add(base + c) = v[k] / s };
        }
    }
}

/// Standardizes each audio column over the row's real text positions, in place:
/// `mean = sum(w[t]) / n` and `var = sum((w[t] - mean)^2) / n` over `t < n = lens[b]`,
/// then `w[t] = (w[t] - mean) / sqrt(max(var, 1e-12))` for every `t < t_len`, padding rows
/// included, as `_dtw_prep_batch` does with its text mask
///
/// One thread per column `(s, b, a)`, `heads * batch * cols` of them, consecutive threads
/// on consecutive `a` so every pass over `t` reads coalesced
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn dtw_norm_f32(
    mut w: DisjointSlice<f32>,
    lens: &[i32],
    columns: u32,
    t_len: u32,
    batch: u32,
    cols: u32,
) {
    let i = thread::index_1d_u32(lc).get();
    require(
        (t_len >= 1)
            & (batch >= 1)
            & (cols >= 1)
            & (columns as usize * t_len as usize <= w.len())
            & (columns % cols == 0)
            & (lens.len() >= batch as usize),
    );
    if i >= columns {
        return;
    }

    let (sb, a) = (i / cols, i % cols);
    let b = sb % batch;
    // SAFETY: b < batch <= lens.len()
    let n = unsafe { *lens.get_unchecked(b as usize) };
    let n = (n.max(0) as u32).min(t_len);
    let base = sb as usize * t_len as usize * cols as usize + a as usize;
    let stride = cols as usize;
    let wp = w.as_mut_ptr();

    let mut sum = 0.0f32;
    for t in 0..n as usize {
        // SAFETY: base + t * cols < columns * t_len <= w.len(); the column is this thread's
        sum += unsafe { *wp.add(base + t * stride) };
    }
    let mean = sum / n as f32;
    let mut q = 0.0f32;
    for t in 0..n as usize {
        // SAFETY: as above
        let d = unsafe { *wp.add(base + t * stride) } - mean;
        // torch squares, then sums: two roundings, so no fused multiply-add here
        q += mul_rn_f32(d, d);
    }
    let std = libm::sqrtf((q / n as f32).max(1e-12));
    for t in 0..t_len as usize {
        // SAFETY: as above
        unsafe {
            let p = wp.add(base + t * stride);
            *p = (*p - mean) / std;
        }
    }
}

/// Reflect padding: index `i - 3` of a row of `n > 3` values, mirrored at both ends
/// without repeating the edge (`F.pad(mode="reflect")`)
#[inline(always)]
fn reflect(i: i32, n: i32) -> usize {
    let j = if i < 0 { -i } else { i };
    (if j >= n { 2 * (n - 1) - j } else { j }) as usize
}

/// The width-7 median filter along the audio axis on every head, then the mean over the
/// heads: `out[b, t, a] = (sum over s of median(w[s, b, t, a - 3 ..= a + 3])) * factor`
/// for `a < width`, the columns `_dtw_prep_batch` keeps (`[:, :, :max_audio_cap]`)
///
/// The filter runs over all `cols` audio positions with reflect padding at both ends, as
/// `median_filter` does before the slice. A median is an exact selection, so the sorting
/// network (the reference's bubble passes) matches torch bit for bit; `factor` is the
/// host's `float(outputs) / inputs`, torch's mean-kernel factor. One thread per output
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn dtw_median_mean_f32(
    mut out: DisjointSlice<f32>,
    w: &[f32],
    heads: u32,
    batch: u32,
    t_len: u32,
    cols: u32,
    width: u32,
    factor: f32,
) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    require(
        (cols as usize > MEDIAN / 2)
            & (t_len >= 1)
            & (width <= cols)
            & (heads as usize * batch as usize * t_len as usize * cols as usize <= w.len()),
    );
    let Some(mut out) = out.element_thread32(t) else {
        return;
    };
    let (bt, a) = (i / width, i % width);
    let (b, tt) = (bt / t_len, bt % t_len);

    let mut sum = 0.0f32;
    for s in 0..heads {
        let row = ((s * batch + b) as usize * t_len as usize + tt as usize) * cols as usize;
        let mut x = [0.0f32; MEDIAN];
        #[unroll]
        for j in 0..MEDIAN {
            let c = reflect(a as i32 + j as i32 - (MEDIAN / 2) as i32, cols as i32);
            // SAFETY: row + c < heads * batch * t_len * cols <= w.len(), checked above
            x[j] = unsafe { *w.get_unchecked(row + c) };
        }
        // the reference's bubble passes, enough of them to place the middle element
        #[unroll]
        for p in 0..MEDIAN / 2 + 1 {
            #[unroll]
            for j in 0..MEDIAN - 1 {
                if j < MEDIAN - p - 1 {
                    let (lo, hi) = (x[j].min(x[j + 1]), x[j].max(x[j + 1]));
                    x[j] = lo;
                    x[j + 1] = hi;
                }
            }
        }
        sum += x[MEDIAN / 2];
    }
    out.write(mul_rn_f32(sum, factor));
}
