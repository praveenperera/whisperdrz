//! ECAPA2's FP16-input 3x3 convs as one implicit GEMM each, over a channels-last copy of
//! the trunk activations, with the bias, ReLU and batch norm epilogue fused
//!
//! The FP16 trunk buffers these read are `[rows][cols][CP]`: frequency rows with a zero
//! border row on each side, time columns as in the FP32 `[rows][channels][cols]` grid (a
//! zero border column on each side, `cols` rounded up), and the channels padded to
//! [`CP`] = 168 with zeros, so each pixel is 21 aligned 16-byte chunks. In that layout the
//! three input pixels a 3x3 conv reads along time for one output column and kernel row
//! (`kw` = 0, 1, 2) are one contiguous run of `3 CP` values, and the run for the next
//! output column starts `CP` values later. So for kernel row `kh` the conv is a GEMM whose
//! A operand is the input read with a row stride of `CP` over a depth of `3 CP`:
//! `A[m][kw CP + c] = x[(i + m + kw) CP + c]`, where input pixel `i + m + 1` sits over
//! output pixel `o + m`. A block stages the strip of `BM + 2` input pixels its outputs
//! need once per kernel row, and `ldmatrix` reads the overlapping rows of A from it, so
//! nothing is unrolled into an im2col buffer and no partial products go through memory.
//!
//! - GEMM: rows m = output pixels (`BM` = 128 a block, 16 a warp), columns n = output
//!   channels (all of them in one block, `NT` 8-wide tiles), depth = 3 kernel rows x 512
//!   (`3 CP` = 504 padded to a multiple of 64; the weights are zero past 504, and the A
//!   fragment reads zeros there)
//! - With stride 1, output pixel `(row, col)` reads input pixels `(row - 1 + kh, col - 1 +
//!   kw)`: its flat index `row * cols + col` plus a constant. So the blocks tile the
//!   interior rows as one flat range of pixels and every tile but the last is full. Border
//!   columns get sums that wrap into the neighbouring rows and are written as zero. With
//!   stride 2 a block takes up to `BM` columns of one output row
//! - Weights `[3 kh][NB][512]`, `NB` = `NT` rounded up to even times 8 rows, row `o`
//!   holding `w[o][c][kh][kw]` at `kw CP + c`, zero for `c >= ci` and `o >= co`
//! - FP16 operands on m16n8k16 tensor cores with FP32 accumulation, one chain per output
//!   in depth order, so the result is the same run to run
//! - Pipeline: 64-deep weight stages, double-buffered with `cp.async`, one barrier per
//!   stage. The strip has one buffer, refilled in place for the next kernel row in three
//!   channel groups, each issued once every warp is past the last stage that reads it
//!
//! Each block writes its tile of the output, border columns included (zero), and zero in
//! channels `co..CP` of a channels-last output; every block zeroes a slice of the two
//! border rows. So the output is a valid input for the next conv

use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::thread::{
    __unroll_config, blockIdx_x, blockIdx_y, gridDim_x, gridDim_y, sync_threads, threadIdx_x,
};
use cuda_device::wmma::ldmatrix_x4_shared_u32;
use cuda_device::{DisjointSlice, DynamicSharedArray, kernel, launch_bounds};

use crate::common::f2h;
use crate::mma_f16::{
    FragA, FragB, SharedTile, cp_async_commit, cp_async_wait_all, cp_async16, mma, pack_f16,
};

/// Channels of a channels-last FP16 trunk buffer: 164 padded to a multiple of 8
pub const CP: usize = 168;
/// Depth per kernel row: `3 CP` padded to a multiple of [`BK`]
pub const KP: usize = 512;
/// Depth of a weight stage
const BK: usize = 64;
/// Weight stages per kernel row
const KSTEPS: u32 = (KP / BK) as u32;
/// Output pixels per block
pub const BM: usize = 128;
const WARPS: usize = BM / 16;
const THREADS: usize = 32 * WARPS;
/// Input pixels of a strip
const STRIP_PIX: usize = BM + 2;
const STRIP_BYTES: usize = STRIP_PIX * CP * 2;
/// 16-byte chunks per pixel
const PIX_CHUNKS: u32 = (CP / 8) as u32;
const _: () = assert!(BM * CP * 2 <= STRIP_BYTES, "the epilogue tile reuses the strip");
// the strip refill schedule in conv_body (channel groups 0..48, 48..112, 112..168 after
// stages 5, 6 and 7, and the zero block for depth 504..512) is worked out for these
const _: () = assert!(CP == 168 && KP == 512 && BK == 64, "redo the strip refill schedule");

/// Output layouts
pub const OUT_NHWC_F16: u32 = 0;
pub const OUT_NCHW_F32: u32 = 1;
pub const OUT_NCHW_F16: u32 = 2;

/// Dynamic shared memory of a conv kernel with `nb` weight rows: the strip, 16 zero
/// bytes, two weight stages, and bias and batch norm as `[4][nb]` floats
pub const fn smem_bytes(nb: usize) -> usize {
    STRIP_BYTES + 16 + 2 * nb * BK * 2 + 4 * nb * 4
}

/// Starts the copy of channel chunks `[LO, LO + W)` of every strip pixel: pixel `i` is
/// input pixel `base + i`, zero-filled outside `[0, pixels)`
///
/// # Safety
/// `x` holds `pixels * CP` values; nothing reads those chunks of `dst` until this thread
/// waits on the copy and a barrier follows
#[inline(always)]
unsafe fn load_strip<const LO: u32, const W: u32>(
    dst: SharedTile<CP>,
    x: *const u16,
    base: i64,
    pixels: i64,
    tid: u32,
) {
    let mut j = tid;
    while j < STRIP_PIX as u32 * W {
        // W is a constant, so these are a multiply and a shift
        let (i, cc) = (j / W, LO + j % W);
        let p = base + i as i64;
        let full = (p >= 0) & (p < pixels);
        let src = if full { x.wrapping_add(p as usize * CP + (cc * 8) as usize) } else { x };
        unsafe { cp_async16(dst.at(i as usize, (cc * 8) as usize), src, full) };
        j += THREADS as u32;
    }
}

/// Starts the copy of weight stage `s` of kernel row `kh`: row `n`, 16-byte chunk `ch`
/// stored at chunk `ch ^ (n % 8)`, so the `ldmatrix` reads of 8 rows hit distinct banks
///
/// # Safety
/// `w` holds `3 * NB * KP` values; nothing reads `dst` until this thread waits on the copy
/// and a barrier follows
#[inline(always)]
unsafe fn load_w<const NB: usize>(dst: SharedTile<BK>, w: *const u16, kh: u32, s: u32, tid: u32) {
    let mut c = tid;
    while c < (NB * BK / 8) as u32 {
        let (n, ch) = (c >> 3, c & 7);
        let row = (kh as usize * NB + n as usize) * KP;
        let src = w.wrapping_add(row + s as usize * BK + ch as usize * 8);
        unsafe { cp_async16(dst.at(n as usize, ((ch ^ (n & 7)) * 8) as usize), src, true) };
        c += THREADS as u32;
    }
}

/// The body of the conv kernels; see the module docs
///
/// # Safety
/// Every thread of a [`THREADS`]-thread block calls it with the launch's own params and
/// [`smem_bytes`]`(NB)` of dynamic shared memory; the check at the top traps on anything
/// that would read or write out of bounds
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn conv_body<const NT: usize, const NP: usize, const NB: usize>(
    out16: &mut DisjointSlice<u16>,
    out32: &mut DisjointSlice<f32>,
    x: &[u16],
    w: &[u16],
    bias: &[f32],
    bn: &[f32],
    (rows_in, cols, width, rows_out, stride, co, mode): (u32, u32, u32, u32, u32, u32, u32),
) {
    let tid = threadIdx_x();
    let (warp, lane) = (tid / 32, tid % 32);
    let (g, q) = (lane / 4, lane % 4);
    let (cols_u, co_u) = (cols as usize, co as usize);
    let out_len = match mode {
        OUT_NHWC_F16 => rows_out as usize * cols_u * CP,
        _ => rows_out as usize * co_u * cols_u,
    };
    let h = rows_out.wrapping_sub(2);
    let flat = stride == 1;
    // stride 1: x = tiles over the flat interior, y = 1; stride 2: x = tiles over a row,
    // y = interior rows
    let tiles = if flat { (h as usize * cols_u).div_ceil(BM) } else { cols_u.div_ceil(BM) };

    // everything the block touches, checked once: one uniform branch to one trap
    let ok = (co >= 1)
        & (co as usize <= NT * 8)
        & ((mode != OUT_NHWC_F16) | (co as usize <= CP))
        & (mode <= OUT_NCHW_F16)
        & ((stride == 1) | (stride == 2))
        & (rows_out >= 3)
        & (h.wrapping_sub(1) * stride + 2 < rows_in)
        & (width + 2 <= cols)
        & (gridDim_x() as usize == tiles)
        & (gridDim_y() == if flat { 1 } else { h })
        & (x.len() >= rows_in as usize * cols_u * CP)
        & (w.len() >= 3 * NB * KP)
        & (bias.len() >= co_u)
        & (bn.len() >= 3 * co_u)
        & (if mode == OUT_NCHW_F32 { out32.len() } else { out16.len() } >= out_len);
    if !ok {
        panic!();
    }
    let o16 = out16.as_mut_ptr();
    let o32 = out32.as_mut_ptr();
    let r = if flat { 0 } else { blockIdx_y() };
    // first output pixel of the tile (flat index, border rows counted), and the end of the
    // range the block may write
    let o_base = (r + 1) * cols + blockIdx_x() * BM as u32;
    let o_end = if flat { (h + 1) * cols } else { (r + 2) * cols };
    // strip pixel i of kernel row kh is input pixel o_base + in_off + kh * cols + i
    let in_off = (r * (stride - 1)) as i64 * cols as i64 - cols as i64 - 1;
    let pixels = rows_in as i64 * cols as i64;
    let xp = x.as_ptr();
    let wp = w.as_ptr();

    let smem = DynamicSharedArray::<u8, 16>::get_raw();
    let zero_ptr = smem.wrapping_add(STRIP_BYTES);
    let stage0_ptr = zero_ptr.wrapping_add(16);
    let stage1_ptr = stage0_ptr.wrapping_add(NB * BK * 2);
    let par = stage1_ptr.wrapping_add(NB * BK * 2) as *mut f32;
    // SAFETY: the dynamic shared memory holds smem_bytes(NB), laid out as above. Stages
    // and strips are selected with `if`, not indexed: a 2-entry array indexed by a loop
    // counter stays on the stack
    let (strip, stage0, stage1) = unsafe {
        (
            SharedTile::<CP>::new(smem as *mut u16),
            SharedTile::<BK>::new(stage0_ptr as *mut u16),
            SharedTile::<BK>::new(stage1_ptr as *mut u16),
        )
    };
    let (strip_u32, zero_u32, sb0, sb1) = unsafe {
        (
            cvta_generic_to_shared_u32(smem as *const u8),
            cvta_generic_to_shared_u32(zero_ptr as *const u8),
            cvta_generic_to_shared_u32(stage0_ptr as *const u8),
            cvta_generic_to_shared_u32(stage1_ptr as *const u8),
        )
    };

    // prologue: the whole strip of kernel row 0 and the first weight stage
    unsafe {
        load_strip::<0, PIX_CHUNKS>(strip, xp, o_base as i64 + in_off, pixels, tid);
        load_w::<NB>(stage0, wp, 0, 0, tid);
        cp_async_commit();
    }
    // the zero block, and bias and batch norm, zero past co so those channels come out zero
    if tid < 16 {
        unsafe { *zero_ptr.add(tid as usize) = 0 };
    }
    let mut i = tid as usize;
    while i < NB {
        let have = i < co_u;
        // SAFETY: i < co inside bias and bn (checked above); i < NB inside the table
        unsafe {
            *par.add(i) = if have { *bias.get_unchecked(i) } else { 0.0 };
            *par.add(NB + i) = if have { *bn.get_unchecked(i) } else { 0.0 };
            *par.add(2 * NB + i) = if have { *bn.get_unchecked(co_u + i) } else { 0.0 };
            *par.add(3 * NB + i) = if have { *bn.get_unchecked(2 * co_u + i) } else { 0.0 };
        }
        i += THREADS;
    }

    // per-lane ldmatrix addresses. A: matrix mi = lane / 8 is rows m0 + (mi % 2) * 8..,
    // depth + (mi / 2) * 8, the A register order; B pair: rows n0 + (mi / 2) * 8.., depth
    // half mi % 2
    let (mi, ri) = (lane / 8, lane % 8);
    let m0 = warp * 16;
    let a_lane = strip_u32 + ((m0 + (mi & 1) * 8 + ri) * CP as u32 + (mi >> 1) * 8) * 2;
    let a_hi = mi >= 2;
    let b_lane = ((mi >> 1) * 8 + ri) * (BK as u32 * 2);
    let b_half = mi & 1;
    // a warp whose 16 pixels all lie past the range skips the math
    let active = o_base + m0 < o_end;

    let mut acc = [[0.0f32; 4]; NT];
    let mut step = 0u32;
    while step < 3 * KSTEPS {
        let kh = step / KSTEPS;
        let s = step % KSTEPS;
        // this stage (and the strip groups it reads) has landed for every thread, and
        // every warp is done with what the loads below overwrite
        unsafe { cp_async_wait_all() };
        sync_threads();
        unsafe {
            if step + 1 < 3 * KSTEPS {
                let next = if step & 1 == 0 { stage1 } else { stage0 };
                load_w::<NB>(next, wp, (step + 1) / KSTEPS, (step + 1) % KSTEPS, tid);
            }
            // the next kernel row's strip a channel group at a time: channels 0..48 are
            // last read in stage 5, 48..112 in stage 6, 112..168 in stage 7; stage 0 reads
            // channels below 64 only, and its wait covers the first two groups
            let next_base = o_base as i64 + in_off + (kh + 1) as i64 * cols as i64;
            if (kh < 2) & (s == 6) {
                load_strip::<0, 6>(strip, xp, next_base, pixels, tid);
            } else if (kh < 2) & (s == 7) {
                load_strip::<6, 8>(strip, xp, next_base, pixels, tid);
            } else if (kh > 0) & (s == 0) {
                load_strip::<14, 7>(strip, xp, next_base - cols as i64, pixels, tid);
            }
            cp_async_commit();
        }

        if active {
            let sb = if step & 1 == 0 { sb0 } else { sb1 };
            for kk in 0..BK / 16 {
                __unroll_config::<0>();
                let k0 = s * BK as u32 + (kk * 16) as u32;
                // depth 504..512 is padding past the strip's last pixel, and the next
                // kernel row's first group may be landing there: those lanes read zeros
                let pad = (kk == BK / 16 - 1) & (s == KSTEPS - 1) & a_hi;
                let addr = if pad { zero_u32 } else { a_lane + k0 * 2 };
                // SAFETY: inside the strip or the zero block, after the barrier above
                let a = FragA::from_bits(unsafe { ldmatrix_x4_shared_u32(addr) });
                let chunk = ((2 * kk) as u32 + b_half) ^ ri;
                for j in 0..NP {
                    __unroll_config::<0>();
                    let addr = sb + b_lane + (j * 16 * BK * 2) as u32 + chunk * 16;
                    // SAFETY: rows below NB of the stage, after the barrier above
                    let rr = unsafe { ldmatrix_x4_shared_u32(addr) };
                    acc[2 * j] = unsafe { mma(acc[2 * j], a, FragB::from_bits([rr[0], rr[1]])) };
                    if 2 * j + 1 < NT {
                        acc[2 * j + 1] =
                            unsafe { mma(acc[2 * j + 1], a, FragB::from_bits([rr[2], rr[3]])) };
                    }
                }
            }
        }
        step += 1;
    }

    // epilogue: bias, ReLU, batch norm; zero on the border columns. Every warp is done
    // with the strip before a channels-last tile is staged in it
    sync_threads();
    let tile = smem as *mut u32;
    for half in 0..2 {
        __unroll_config::<0>();
        let m = m0 + (half * 8) as u32 + g;
        let o = o_base + m;
        let (row, p) = (o / cols, o % cols);
        let border = (p == 0) | (p > width);
        for nt in 0..NT {
            __unroll_config::<0>();
            let c = nt * 8 + 2 * q as usize;
            let mut v = [0.0f32; 2];
            for e in 0..2 {
                __unroll_config::<0>();
                let ch = c + e;
                // SAFETY: ch < NB, inside the table
                let (bi, mean, k, b) = unsafe {
                    (*par.add(ch), *par.add(NB + ch), *par.add(2 * NB + ch), *par.add(3 * NB + ch))
                };
                let t = acc[nt][half * 2 + e] + bi;
                // torch's ReLU: NaN kept
                let t = if t < 0.0 { 0.0 } else { t };
                v[e] = if border { 0.0 } else { (t - mean) * k + b };
            }
            // SAFETY: the staged tile lies in the strip; global stores lie inside the
            // output for o < o_end (checked above)
            unsafe {
                if mode == OUT_NHWC_F16 {
                    if c < CP {
                        *tile.add((m as usize * CP + c) / 2) = pack_f16(v[0], v[1]);
                    }
                } else if o < o_end {
                    for e in 0..2 {
                        __unroll_config::<0>();
                        if c + e < co_u {
                            let at = (row as usize * co_u + c + e) * cols_u + p as usize;
                            if mode == OUT_NCHW_F32 {
                                *o32.add(at) = v[e];
                            } else {
                                *o16.add(at) = f2h(v[e]);
                            }
                        }
                    }
                }
            }
        }
    }
    if mode == OUT_NHWC_F16 {
        sync_threads();
        // the tile's valid pixels are one contiguous run of the output
        let valid = (o_end.min(o_base + BM as u32) - o_base) * PIX_CHUNKS;
        let src = smem as *const [u32; 4];
        let dst = (o16 as *mut [u32; 4]).wrapping_add(o_base as usize * PIX_CHUNKS as usize);
        let mut c = tid;
        while c < valid {
            // SAFETY: inside the staged tile, and inside the output as o < o_end (checked
            // above); both 16-byte aligned
            unsafe { *dst.add(c as usize) = *src.add(c as usize) };
            c += THREADS as u32;
        }
    }

    // this block's slice of the two border rows
    let blocks = (gridDim_x() * gridDim_y()) as usize * THREADS;
    let id = (blockIdx_y() * gridDim_x() + blockIdx_x()) as usize * THREADS + tid as usize;
    let last = (rows_out - 1) as usize;
    if mode == OUT_NHWC_F16 {
        let row_words = cols_u * CP / 2;
        let mut i = id;
        while i < 2 * row_words {
            let at = if i < row_words { i } else { last * row_words + i - row_words };
            // SAFETY: inside the first or last output row, 4-byte aligned
            unsafe { *(o16 as *mut u32).add(at) = 0 };
            i += blocks;
        }
    } else {
        let row_len = co_u * cols_u;
        let mut i = id;
        while i < 2 * row_len {
            let at = if i < row_len { i } else { last * row_len + i - row_len };
            // SAFETY: inside the first or last output row
            unsafe {
                if mode == OUT_NCHW_F32 {
                    *o32.add(at) = 0.0;
                } else {
                    *o16.add(at) = 0;
                }
            }
            i += blocks;
        }
    }
}

macro_rules! conv_kernels {
    ($($name:ident = ($nt:literal, $np:literal, $nb:literal);)*) => {$(
        /// A 3x3 conv (padding 1, stride `stride` along rows) of a channels-last FP16 trunk
        /// buffer `x` (`[rows_in][cols][CP]`) into `rows_out` padded output rows, with bias,
        /// ReLU and batch norm (`bn` = `[mean; k; b]` over `co`); see the module docs.
        /// `mode` picks the output: channels-last FP16 into `out16` (`[rows_out][cols][CP]`),
        /// FP32 `[rows_out][co][cols]` into `out32`, or the same in FP16 into `out16`.
        /// Grid: stride 1, x = `ceil((rows_out - 2) * cols / BM)`, y = 1; stride 2, x =
        /// `ceil(cols / BM)`, y = `rows_out - 2`. Block: 256 threads with `smem_bytes(NB)`
        /// of dynamic shared memory
        #[kernel]
        #[launch_bounds(256, 1)]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            mut out16: DisjointSlice<u16>,
            mut out32: DisjointSlice<f32>,
            x: &[u16],
            w: &[u16],
            bias: &[f32],
            bn: &[f32],
            rows_in: u32,
            cols: u32,
            width: u32,
            rows_out: u32,
            stride: u32,
            co: u32,
            mode: u32,
        ) {
            // SAFETY: the body checks every bound it relies on
            unsafe {
                conv_body::<$nt, $np, $nb>(
                    &mut out16,
                    &mut out32,
                    x,
                    w,
                    bias,
                    bn,
                    (rows_in, cols, width, rows_out, stride, co, mode),
                )
            };
        }
    )*};
}

conv_kernels! {
    ecapa_conv3x3_nhwc_n168 = (21, 11, 176);
}

/// Pixels per block of [`ecapa_scale_residual_nhwc`]
pub const SR_COLS: usize = 32;

/// `out = x * s[channel] + r` over a trunk buffer `[rows][channels][cols]` (pad 1, zero
/// border), written as FP32 to `out` and as a channels-last FP16 copy (round to nearest
/// even) to `out_h` (`[rows][cols][CP]`, zero in channels `channels..CP`), the input of
/// the next block's [`ecapa_conv3x3_nhwc_n168`]. Empty `out`, `s` or `r` skip the FP32
/// write, the scale and the residual, so with all three empty it converts an FP32 buffer.
/// Grid: x = `ceil(cols / SR_COLS)`, y = `rows`. Block: 256 threads with
/// `SR_COLS * CP * 2` bytes of dynamic shared memory
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ecapa_scale_residual_nhwc(
    mut out: DisjointSlice<f32>,
    mut out_h: DisjointSlice<u16>,
    x: &[f32],
    s: &[f32],
    r: &[f32],
    rows: u32,
    channels: u32,
    cols: u32,
    width: u32,
) {
    let tid = threadIdx_x();
    let (h, t0) = (blockIdx_y(), blockIdx_x() * SR_COLS as u32);
    let len = rows as usize * channels as usize * cols as usize;
    let ok = (channels as usize <= CP)
        & (width + 2 <= cols)
        & (h < rows)
        & (t0 < cols)
        & (x.len() >= len)
        & (out.is_empty() | (out.len() >= len))
        & (out_h.len() >= rows as usize * cols as usize * CP)
        & (s.is_empty() | (s.len() >= channels as usize))
        & (r.is_empty() | (r.len() >= len));
    if !ok {
        panic!();
    }
    let border_row = (h == 0) | (h == rows - 1);
    let tile = DynamicSharedArray::<u8, 16>::get_raw() as *mut u16;
    let o = out.as_mut_ptr();

    // one channel's SR_COLS pixels per warp pass: coalesced FP32 reads and writes
    let mut i = tid;
    while i < CP as u32 * SR_COLS as u32 {
        let (c, j) = (i / SR_COLS as u32, i % SR_COLS as u32);
        let w = t0 + j;
        let mut v = 0.0;
        if (c < channels) & (w < cols) {
            let at = (h as usize * channels as usize + c as usize) * cols as usize + w as usize;
            if !border_row & (w != 0) & (w <= width) {
                // SAFETY: at < len, inside x, s and r when they are not empty (checked above)
                unsafe {
                    v = *x.get_unchecked(at);
                    if !s.is_empty() {
                        v *= *s.get_unchecked(c as usize);
                    }
                    if !r.is_empty() {
                        v += *r.get_unchecked(at);
                    }
                }
            }
            if !out.is_empty() {
                // SAFETY: at < len, inside out (checked above)
                unsafe { *o.add(at) = v };
            }
        }
        // SAFETY: j < SR_COLS and c < CP, inside the tile
        unsafe { *tile.add(j as usize * CP + c as usize) = f2h(v) };
        i += THREADS as u32;
    }
    sync_threads();

    // the tile's pixels are one contiguous run of the copy
    let valid = (cols - t0).min(SR_COLS as u32) * PIX_CHUNKS;
    let src = tile as *const [u32; 4];
    let dst = (out_h.as_mut_ptr() as *mut [u32; 4])
        .wrapping_add((h as usize * cols as usize + t0 as usize) * PIX_CHUNKS as usize);
    let mut c = tid;
    while c < valid {
        // SAFETY: inside the tile, and inside out_h (checked above); both 16-byte aligned
        unsafe { *dst.add(c as usize) = *src.add(c as usize) };
        c += THREADS as u32;
    }
}
