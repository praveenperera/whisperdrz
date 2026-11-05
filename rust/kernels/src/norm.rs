//! LayerNorm on FP16 rows with FP32 statistics, one warp per row
//!
//! torch's FP16 LayerNorm widens each value to FP32, takes the row mean and population
//! variance in FP32 (Welford), and writes `f16(w * (rsqrt(var + eps) * (x - mean)) + b)`.
//! These kernels take the same statistics in two passes over registers (sum, then sum of
//! squared deviations), equal in real arithmetic, and apply the same affine form. A warp
//! holds its row in registers as packed halves, eight per 16-byte chunk, so rows of up to
//! `32 * MAX_CHUNKS * 8 = 2048` values need no shared memory or barriers

use cuda_device::thread::{__unroll_config, blockIdx_x, threadIdx_x};
use cuda_device::{DisjointSlice, kernel, launch_contract, warp};

use crate::pointwise::H8;

// torch's LayerNorm takes `rsqrt` from the CUDA math library, which is this libdevice
// function; libm has no rsqrtf for cuda-oxide to map
unsafe extern "C" {
    fn __nv_rsqrtf(x: f32) -> f32;
}

/// 16-byte chunks per lane: rows up to 2048 halves
const MAX_CHUNKS: usize = 8;
/// Rows per 128-thread block
const ROWS_PER_BLOCK: u32 = 4;

/// Traps when the host's geometry doesn't fit the buffers it passed, as a failed slice
/// index would
#[inline(always)]
fn require(ok: bool) {
    if !ok {
        panic!();
    }
}

#[inline(always)]
fn warp_sum(mut v: f32) -> f32 {
    v += warp::shuffle_xor_f32(v, 16);
    v += warp::shuffle_xor_f32(v, 8);
    v += warp::shuffle_xor_f32(v, 4);
    v += warp::shuffle_xor_f32(v, 2);
    v += warp::shuffle_xor_f32(v, 1);
    v
}

/// The row and lane this thread works on, after checking the buffers once for the whole
/// launch, or `None` for the spare warps of the last block
#[inline(always)]
fn row_of(rows: u32, cols8: u32, lens: [usize; 3], w: &[H8], b: &[H8]) -> Option<(usize, u32)> {
    let n = rows as usize * cols8 as usize;
    require(
        (cols8 >= 1)
            & (cols8 as usize <= 32 * MAX_CHUNKS)
            & (lens[0] >= n)
            & (lens[1] >= n)
            & (lens[2] >= n)
            & (w.len() >= cols8 as usize)
            & (b.len() >= cols8 as usize),
    );
    let row = blockIdx_x() * ROWS_PER_BLOCK + threadIdx_x() / 32;
    (row < rows).then_some((row as usize * cols8 as usize, threadIdx_x() % 32))
}

/// This lane's chunks of the LayerNorm weight and bias, chunk `lane + 32 i` at `i`, for
/// a kernel that loads them before its row
///
/// # Safety
/// `w` and `b` hold `cols8` chunks
#[inline(always)]
unsafe fn affine(
    lane: u32,
    cols8: u32,
    w: &[H8],
    b: &[H8],
) -> ([H8; MAX_CHUNKS], [H8; MAX_CHUNKS]) {
    let mut wv = [H8([0; 4]); MAX_CHUNKS];
    let mut bv = [H8([0; 4]); MAX_CHUNKS];
    for i in 0..MAX_CHUNKS {
        __unroll_config::<0>();
        let c = (lane + 32 * i as u32) as usize;
        if c < cols8 as usize {
            // SAFETY: c < cols8 <= w.len(), b.len()
            unsafe {
                wv[i] = *w.get_unchecked(c);
                bv[i] = *b.get_unchecked(c);
            }
        }
    }
    (wv, bv)
}

/// Normalizes a row held as `v[i]` = chunk `lane + 32 i` (chunks at or past `cols8` are
/// zero) with the weight and bias chunks `affine(i, c)` gives for chunk `c`, and stores
/// chunk `c` through `store(c, value)`
///
/// # Safety
/// The whole warp calls it together; `affine` is safe to call for every `c < cols8`
#[inline(always)]
unsafe fn normalize(
    v: &[H8; MAX_CHUNKS],
    lane: u32,
    cols8: u32,
    affine: impl Fn(usize, usize) -> ([f32; 8], [f32; 8]),
    eps: f32,
    mut store: impl FnMut(usize, H8),
) {
    let n = (cols8 * 8) as f32;
    let mut s = 0.0f32;
    for i in 0..MAX_CHUNKS {
        __unroll_config::<0>();
        let x = v[i].unpack();
        for e in 0..8 {
            __unroll_config::<0>();
            s += x[e];
        }
    }
    let mean = warp_sum(s) / n;

    let mut q = 0.0f32;
    for i in 0..MAX_CHUNKS {
        __unroll_config::<0>();
        if lane + 32 * (i as u32) < cols8 {
            let x = v[i].unpack();
            for e in 0..8 {
                __unroll_config::<0>();
                let d = x[e] - mean;
                q += d * d;
            }
        }
    }
    // SAFETY: a libdevice function, pure
    let rstd = unsafe { __nv_rsqrtf(warp_sum(q) / n + eps) };

    for i in 0..MAX_CHUNKS {
        __unroll_config::<0>();
        let c = (lane + 32 * i as u32) as usize;
        if c < cols8 as usize {
            let x = v[i].unpack();
            let (wc, bc) = affine(i, c);
            let mut y = [0.0f32; 8];
            for e in 0..8 {
                __unroll_config::<0>();
                y[e] = wc[e] * (rstd * (x[e] - mean)) + bc[e];
            }
            store(c, H8::pack(y));
        }
    }
}

/// LayerNorm over rows of `cols8 * 8` FP16 values: `out[r] = LN(x[r]) * w + b`
///
/// `x` and `out` are `[rows, cols8]` in 8-value chunks, `w` and `b` `[cols8]`; `cols8 <=
/// 256`. Grid `ceil(rows / 4)` blocks of 128 threads, one warp per row
#[kernel]
#[launch_contract(domain = 1, block = (128, 1, 1))]
pub fn layernorm_f16(
    mut out: DisjointSlice<H8>,
    x: &[H8],
    w: &[H8],
    b: &[H8],
    rows: u32,
    cols8: u32,
    eps: f32,
) {
    let Some((base, lane)) = row_of(rows, cols8, [x.len(), out.len(), out.len()], w, b) else {
        return;
    };
    let mut v = [H8([0; 4]); MAX_CHUNKS];
    #[unroll]
    for i in 0..MAX_CHUNKS {
        let c = (lane + 32 * i as u32) as usize;
        if c < cols8 as usize {
            // SAFETY: base + c < rows * cols8 <= x.len(), checked in row_of
            v[i] = unsafe { *x.get_unchecked(base + c) };
        }
    }
    let op = out.as_mut_ptr();
    // SAFETY: the warp runs it together; each chunk of this row is stored by one lane
    // only, inside the checked rows * cols8 <= out.len(); c < cols8 <= w.len(), b.len()
    unsafe {
        normalize(
            &v,
            lane,
            cols8,
            |_, c| (w.get_unchecked(c).unpack(), b.get_unchecked(c).unpack()),
            eps,
            |c, y| *op.add(base + c) = y,
        )
    };
}

/// The residual add before a LayerNorm, fused: `x[r] = f16(x[r] + add[r])`, then
/// `out[r] = LN(x[r]) * w + b` of the rounded sum, as torch's separate add and
/// LayerNorm compute it. Shapes as [`layernorm_f16`]; `x` is updated in place
#[kernel]
#[launch_contract(domain = 1, block = (128, 1, 1))]
pub fn add_layernorm_f16(
    mut x: DisjointSlice<H8>,
    add: &[H8],
    mut out: DisjointSlice<H8>,
    w: &[H8],
    b: &[H8],
    rows: u32,
    cols8: u32,
    eps: f32,
) {
    let Some((base, lane)) = row_of(rows, cols8, [x.len(), add.len(), out.len()], w, b) else {
        return;
    };
    let xp = x.as_mut_ptr();
    // the weight and bias, then the row, all loaded before the first store of x, which
    // the compiler cannot move a later load above: loaded next to each store, every
    // chunk waited out its own memory round trip, and a one-row add and LayerNorm in a
    // decode step is nothing but that latency
    // SAFETY: row_of checked w and b against cols8
    let (wv, bv) = unsafe { affine(lane, cols8, w, b) };
    let mut xs = [H8([0; 4]); MAX_CHUNKS];
    let mut ds = [H8([0; 4]); MAX_CHUNKS];
    #[unroll]
    for i in 0..MAX_CHUNKS {
        let c = (lane + 32 * i as u32) as usize;
        if c < cols8 as usize {
            // SAFETY: base + c < rows * cols8, inside x and add (checked in row_of)
            unsafe {
                xs[i] = *xp.add(base + c);
                ds[i] = *add.get_unchecked(base + c);
            }
        }
    }
    let mut v = [H8([0; 4]); MAX_CHUNKS];
    #[unroll]
    for i in 0..MAX_CHUNKS {
        let c = (lane + 32 * i as u32) as usize;
        if c < cols8 as usize {
            // SAFETY: as above; the chunk is this lane's alone, read then written by it
            // only
            unsafe {
                let (a, d) = (xs[i].unpack(), ds[i].unpack());
                let mut s = [0.0f32; 8];
                #[unroll]
                for e in 0..8 {
                    s[e] = a[e] + d[e];
                }
                let sum = H8::pack(s);
                *xp.add(base + c) = sum;
                v[i] = sum;
            }
        }
    }
    let op = out.as_mut_ptr();
    // SAFETY: as in layernorm_f16
    unsafe {
        normalize(
            &v,
            lane,
            cols8,
            |i, _| (wv[i].unpack(), bv[i].unpack()),
            eps,
            |c, y| *op.add(base + c) = y,
        )
    };
}
