//! Small-M linears for the decode step: `y = x @ w^T + b` for up to 16 rows, FP16 operands
//! on m16n8k16 tensor cores with FP32 sums, optionally followed by GELU
//!
//! At the decode step's small buckets a linear is a weight stream: every weight byte is read
//! once and used against a handful of rows, so the limit is how many bytes are in flight, not
//! the math. Each warp takes 8 output features by `CH` 32-deep chunks of the inner dimension
//! and requests all of its weights, its rows of `x` and its biases into registers before it
//! waits on any of them; with the grid co-resident, a launch has the whole matrix in flight
//! at once. A block is 4 warps: `4 / wk` feature tiles by `wk` warps along the depth.
//!
//! The weights are the MMA's B operand (8 features a fragment) and the rows of `x` its A
//! operand (16 rows a fragment, rows past 8 zero and never loaded in the `r8` kernels). The
//! depth of each pair of MMAs is permuted: lane `(g, t)` loads 16 contiguous bytes, depth
//! `8t..8t + 8` of a 32-deep chunk, of weight row `g` and input rows `g` and `g + 8`, and the
//! first MMA takes depth `8t..8t + 4` where the PTX layout puts `2t, 2t + 1, 2t + 8, 2t + 9`,
//! the second `8t + 4..8t + 8`. A and B use the same permutation, so the sum is the same set
//! of products.
//!
//! The `wk` warps over one feature tile sum their accumulators through shared memory in warp
//! order. With `splits > 1` blocks along y split the depth further: each writes its FP32
//! partial, and the last block of a feature tile to finish, found with an integer counter,
//! sums the partials in split order 0, 1, .. and resets the counter. The bias is added in
//! FP32 and the result rounded to FP16 once, as cuBLASLt's bias epilogue does; with `gelu`
//! the rounded value then goes through GELU and is rounded again, which is what the separate
//! `gelu_f16x8` pass computes. No float atomics and a fixed order everywhere, so the output
//! is the same run to run
//!
//! Only the variants the decode routing uses are built (`small_m::route` in
//! whisperdrz-native). There are none for 32 to 56 rows: kernels of this design with the
//! weights as the A operand and `x` staged in shared memory ran 0.63 to 0.85x of cuBLASLt's
//! split-K kernels at every bucket from 42 rows up on an RTX 5090

use cuda_device::thread::{
    __unroll_config, blockIdx_x, blockIdx_y, blockIdx_z, gridDim_x, sync_threads, threadIdx_x,
};
use cuda_device::{DisjointSlice, DynamicSharedArray, kernel, launch_bounds, ptx_asm};

use crate::common::{f2h, h2f};
use crate::mma_f16::{FragA, FragB, mma};
use crate::pointwise::gelu;

/// Threads per block: 4 warps, `wn * wk == 4`
pub const THREADS: usize = 128;
const WARPS: usize = THREADS / 32;

/// A 16-byte read-only global load for the weights, each byte read exactly once (the
/// sm_80 PTX ISA 7.0 has no `L1::no_allocate`)
///
/// # Safety
/// `p` is a 16-byte aligned global address with 16 readable bytes
#[inline(always)]
unsafe fn ld_stream(p: *const u16) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %4; ld.global.nc.v4.u32 {%0, %1, %2, %3}, [g]; }",
            out("=r") a,
            out("=r") b,
            out("=r") c,
            out("=r") d,
            in("l") p as u64,
        );
    }
    [a, b, c, d]
}

/// A 16-byte global load of four floats from L2, for the partials other blocks wrote in
/// this launch
///
/// # Safety
/// `p` is a 16-byte aligned global address with 16 readable bytes
#[inline(always)]
unsafe fn ld_cg_f32x4(p: *const f32) -> [f32; 4] {
    let (a, b, c, d): (f32, f32, f32, f32);
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %4; ld.global.cg.v4.f32 {%0, %1, %2, %3}, [g]; }",
            out("=f") a,
            out("=f") b,
            out("=f") c,
            out("=f") d,
            in("l") p as u64,
        );
    }
    [a, b, c, d]
}

/// A 16-byte global store of four floats that skips L1
///
/// # Safety
/// `p` is a 16-byte aligned global address with 16 writable bytes
#[inline(always)]
unsafe fn st_cg_f32x4(p: *mut f32, v: [f32; 4]) {
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %0; st.global.cg.v4.f32 [g], {%1, %2, %3, %4}; }",
            in("l") p as u64,
            in("f") v[0],
            in("f") v[1],
            in("f") v[2],
            in("f") v[3],
            clobber("memory"),
        );
    }
}

/// `fence.acq_rel.gpu`: this thread's earlier global accesses are ordered GPU-wide before
/// its later ones
#[inline(always)]
fn fence_gpu() {
    unsafe { ptx_asm!("fence.acq_rel.gpu;", clobber("memory")) };
}

/// `atomicAdd(p, 1)` at GPU scope, returning the old value
///
/// # Safety
/// `p` is a 4-byte aligned global address only accessed atomically during the launch
#[inline(always)]
unsafe fn atomic_inc_gpu(p: *mut u32) -> u32 {
    let old: u32;
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %1; atom.acq_rel.gpu.global.add.u32 %0, [g], 1; }",
            out("=r") old,
            in("l") p as u64,
            clobber("memory"),
        );
    }
    old
}

/// A plain 4-byte global store at GPU scope, the counter reset
///
/// # Safety
/// `p` is a 4-byte aligned global address
#[inline(always)]
unsafe fn st_relaxed_u32(p: *mut u32, v: u32) {
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %0; st.relaxed.gpu.global.u32 [g], %1; }",
            in("l") p as u64,
            in("r") v,
            clobber("memory"),
        );
    }
}

/// The launch's sizes and strides, all in elements
#[derive(Clone, Copy)]
struct Dims {
    rows: usize,
    k: usize,
    n: usize,
    ldx: usize,
    ldy: usize,
    wk: usize,
    splits: usize,
    gelu: bool,
    /// Per `blockIdx.z` offsets of `x`, `w`, `b` and `y`
    xz: usize,
    wz: usize,
    bz: usize,
    yz: usize,
}

/// Up to 8 partials loaded before any is added, so a split reduction waits for one L2 round
/// trip per 8 splits rather than one per split
const RED_BATCH: usize = 8;

/// Sums `splits` partial fragments at `p`, `stride` floats apart, in split order
///
/// # Safety
/// Each of the `splits` 16-byte loads lies inside the partials, written in this launch before
/// the counter add the caller acquired
#[inline(always)]
unsafe fn sum_splits(p: *const f32, stride: usize, splits: usize) -> [f32; 4] {
    let mut sum = [0.0f32; 4];
    let mut s = 0;
    while s < splits {
        let mut v = [[0.0f32; 4]; RED_BATCH];
        for j in 0..RED_BATCH {
            __unroll_config::<0>();
            if s + j < splits {
                v[j] = unsafe { ld_cg_f32x4(p.wrapping_add((s + j) * stride)) };
            }
        }
        for j in 0..RED_BATCH {
            __unroll_config::<0>();
            if s + j < splits {
                for e in 0..4 {
                    __unroll_config::<0>();
                    sum[e] += v[j][e];
                }
            }
        }
        s += RED_BATCH;
    }
    sum
}

/// Rounds, optionally GELUs and stores one narrow accumulator: row `row`, features `f` and
/// `f + 1` as one 4-byte store
///
/// # Safety
/// `row < rows`, and the features lie inside `y`; `f` is even and `y` 4-byte aligned
#[inline(always)]
unsafe fn store_pair(y: *mut u16, d: &Dims, row: usize, f: usize, (v0, v1): (f32, f32)) {
    let (mut h0, mut h1) = (f2h(v0), f2h(v1));
    if d.gelu {
        h0 = f2h(gelu(h2f(h0)));
        h1 = f2h(gelu(h2f(h1)));
    }
    unsafe { *(y.add(row * d.ldy + f) as *mut u32) = h0 as u32 | (h1 as u32) << 16 };
}

/// The body of the `linear_n8_*` kernels: up to 16 rows (8 with `R16` false), `CH` 32-deep
/// chunks per warp, 8 features a warp tile
///
/// The weights are the MMA's B operand here (8 features a fragment) and the rows of `x` its A
/// operand, so a 16-row fragment covers every row. Both operands go straight to registers:
/// no shared staging, no barrier before the MMAs. With `R16` false rows 8..16 of the A
/// fragment are zero and never loaded
///
/// # Safety
/// Every thread of a 128-thread block calls it with the launch's own params; the check at
/// the top traps on anything that would read or write out of bounds
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn narrow_body<const R16: bool, const CH: usize>(
    mut y: DisjointSlice<u16>,
    mut part: DisjointSlice<f32>,
    mut counters: DisjointSlice<u32>,
    x: &[u16],
    w: &[u16],
    b: &[u16],
    d: Dims,
) {
    let tid = threadIdx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let (g, t) = (lane / 4, lane % 4);
    // wk is 1, 2 or 4 (checked below), so shifts stand in for runtime divisions, which compile
    // to calls into a division routine
    let wk_log = (d.wk as u32).trailing_zeros().min(2);
    let wn = WARPS >> wk_log;
    let (wn_i, wk_i) = (warp & (wn - 1), warp >> (2 - wk_log));
    let ks = d.wk * CH * 32;
    let tiles = gridDim_x() as usize;
    let tile = blockIdx_x() as usize;
    let split = blockIdx_y() as usize;
    let z = blockIdx_z() as usize;
    let f_cta = tile * 8 * wn;
    let max_rows = if R16 { 16 } else { 8 };

    let ok = ((d.wk == 1) | (d.wk == 2) | (d.wk == 4))
        & (d.rows >= 1)
        & (d.rows <= max_rows)
        & (d.splits >= 1)
        & (split < d.splits)
        & (d.k == d.splits * ks)
        & (d.n == tiles * 8 * wn)
        & (d.ldx >= d.k)
        & (d.ldy >= d.n)
        & (d.ldx % 8 == 0)
        & (d.ldy % 2 == 0)
        & (d.yz % 2 == 0)
        & (d.k % 8 == 0)
        & (x.len() >= z * d.xz + d.rows.saturating_sub(1) * d.ldx + d.k)
        & (w.len() >= z * d.wz + d.n * d.k)
        & (b.len() >= z * d.bz + d.n)
        & (y.len() >= z * d.yz + d.rows.saturating_sub(1) * d.ldy + d.n)
        & ((d.splits == 1)
            | ((part.len() >= ((z + 1) * tiles * d.splits) * wn * 128)
                & (counters.len() >= (z + 1) * tiles)));
    if !ok {
        panic!();
    }

    let xp = x.as_ptr().wrapping_add(z * d.xz);
    let wp = w.as_ptr().wrapping_add(z * d.wz);
    let bp = b.as_ptr().wrapping_add(z * d.bz);
    let yp = y.as_mut_ptr().wrapping_add(z * d.yz);

    // every load this warp makes, issued before anything waits: weights of features f_w + g,
    // rows g (and g + 8) of x, both at depth k_w + 32c + 8t.., and the bias of the two
    // features this lane's accumulators hold
    let f_w = f_cta + 8 * wn_i;
    let k_w = split * ks + wk_i * CH * 32;
    let w0 = wp.wrapping_add((f_w + g) * d.k + k_w + 8 * t);
    let x0 = xp.wrapping_add(g * d.ldx + k_w + 8 * t);
    let x1 = x0.wrapping_add(8 * d.ldx);
    let (has0, has1) = (g < d.rows, R16 && g + 8 < d.rows);
    let mut wv = [[0u32; 4]; CH];
    let mut xa = [[0u32; 4]; CH];
    let mut xb = [[0u32; 4]; CH];
    for c in 0..CH {
        __unroll_config::<0>();
        // SAFETY: features below n and depth below k lie inside w, rows below `rows` inside x
        // (checked above); every offset is a multiple of 8 halves
        unsafe {
            wv[c] = ld_stream(w0.add(c * 32));
            if has0 {
                xa[c] = *(x0.add(c * 32) as *const [u32; 4]);
            }
            if has1 {
                xb[c] = *(x1.add(c * 32) as *const [u32; 4]);
            }
        }
    }
    // SAFETY: f_w + 2t + 1 < n (checked above)
    let bias = unsafe { (h2f(*bp.add(f_w + 2 * t)), h2f(*bp.add(f_w + 2 * t + 1))) };

    let mut acc = [0.0f32; 4];
    for c in 0..CH {
        __unroll_config::<0>();
        let a0 = FragA::from_bits([xa[c][0], xb[c][0], xa[c][1], xb[c][1]]);
        let a1 = FragA::from_bits([xa[c][2], xb[c][2], xa[c][3], xb[c][3]]);
        acc = unsafe { mma(acc, a0, FragB::from_bits([wv[c][0], wv[c][1]])) };
        acc = unsafe { mma(acc, a1, FragB::from_bits([wv[c][2], wv[c][3]])) };
    }

    let smem = DynamicSharedArray::<u8, 16>::get_raw();
    let red = smem as *mut [f32; 4];
    let flag = smem.wrapping_add((WARPS - 1) * 32 * 16) as *mut u32;
    if d.wk > 1 {
        if wk_i > 0 {
            // SAFETY: inside the reduction area: (wk_i - 1) * wn + wn_i < WARPS - 1
            unsafe { *red.add(((wk_i - 1) * wn + wn_i) * 32 + lane) = acc };
        }
        sync_threads();
        if wk_i == 0 {
            let mut j = 1;
            while j < d.wk {
                let v = unsafe { *red.add(((j - 1) * wn + wn_i) * 32 + lane) };
                for e in 0..4 {
                    __unroll_config::<0>();
                    acc[e] += v[e];
                }
                j += 1;
            }
        }
    }

    let slot = z * tiles + tile;
    let pp = part.as_mut_ptr();
    if d.splits > 1 {
        if wk_i == 0 {
            let at = ((slot * d.splits + split) * wn + wn_i) * 128 + lane * 4;
            // SAFETY: inside part (checked above), 16-byte aligned
            unsafe { st_cg_f32x4(pp.wrapping_add(at), acc) };
        }
        fence_gpu();
        sync_threads();
        let cp = counters.as_mut_ptr().wrapping_add(slot);
        if tid == 0 {
            // SAFETY: slot < counters.len() (checked above)
            let old = unsafe { atomic_inc_gpu(cp) };
            unsafe { *flag = (old as usize == d.splits - 1) as u32 };
        }
        sync_threads();
        if unsafe { *flag } == 0 {
            return;
        }
        fence_gpu();
        if tid == 0 {
            // SAFETY: as for the add; the next launch on this stream starts after this one
            unsafe { st_relaxed_u32(cp, 0) };
        }
        if wk_i == 0 {
            let at = (slot * d.splits * wn + wn_i) * 128 + lane * 4;
            // SAFETY: the tile's partials, written by this launch's blocks
            acc = unsafe { sum_splits(pp.wrapping_add(at), wn * 128, d.splits) };
        }
    }

    if wk_i == 0 {
        let f = f_w + 2 * t;
        // SAFETY: rows below `rows`, features below n, f even
        unsafe {
            if has0 {
                store_pair(yp, &d, g, f, (acc[0] + bias.0, acc[1] + bias.1));
            }
            if has1 {
                store_pair(yp, &d, g + 8, f, (acc[2] + bias.0, acc[3] + bias.1));
            }
        }
    }
}

macro_rules! narrow_kernels {
    ($($name:ident = ($r16:literal, $ch:literal);)*) => {$(
        /// `y = x @ w^T + b` (then GELU when `gelu != 0`) for up to 16 rows (8 for the `r8`
        /// kernels); see [`narrow_body`]. Grid: x = feature tiles of `8 * (4 / wk)`, y =
        /// `splits`, z = batch. Block: 128 threads with `narrow_smem_bytes()` of dynamic
        /// shared memory. `x` `[rows, ldx]`, `w` `[n, k]`, `y` `[rows, ldy]` with `ldy` and `yz` even, each
        /// batch entry `xz`, `wz`, `bz` and `yz` elements on;
        /// with `splits > 1`, `part` holds `batch * tiles * splits * (4 / wk) * 128` floats
        #[kernel]
        #[launch_bounds(128)]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            y: DisjointSlice<u16>,
            part: DisjointSlice<f32>,
            counters: DisjointSlice<u32>,
            x: &[u16],
            w: &[u16],
            b: &[u16],
            rows: u32,
            k: u32,
            n: u32,
            ldx: u32,
            ldy: u32,
            wk: u32,
            splits: u32,
            gelu: u32,
            xz: u32,
            wz: u32,
            bz: u32,
            yz: u32,
        ) {
            let d = Dims {
                rows: rows as usize,
                k: k as usize,
                n: n as usize,
                ldx: ldx as usize,
                ldy: ldy as usize,
                wk: wk as usize,
                splits: splits as usize,
                gelu: gelu != 0,
                xz: xz as usize,
                wz: wz as usize,
                bz: bz as usize,
                yz: yz as usize,
            };
            // SAFETY: the body checks every bound it relies on
            unsafe { narrow_body::<$r16, $ch>(y, part, counters, x, w, b, d) };
        }
    )*};
}

/// Dynamic shared memory of the narrow kernels: the cross-warp reduction area and a flag
pub const fn narrow_smem_bytes() -> usize {
    (WARPS - 1) * 32 * 16 + 16
}

narrow_kernels! {
    linear_n8_r8_c5 = (false, 5);
    linear_n8_r16_c10 = (true, 10);
}
