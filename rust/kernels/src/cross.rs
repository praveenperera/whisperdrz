//! Decoder cross-attention over the encoder output itself, with the key and value
//! projections folded out of the attention
//!
//! Whisper's cross-attention key has no bias and softmax weights sum to 1, so per head `h`
//! `softmax(scale q_h (xa W_k,h^T)^T) (xa W_v,h^T + b_v,h) = softmax(q'_h xa^T) xa W_v,h^T +
//! b_v,h` with `q'_h = scale q_h W_k,h`. The host runs the two small projections as batched
//! GEMMs; the kernels here run the middle: for each token, its 20 heads become 20 query rows
//! of one 1280-wide head over the token's slot's encoder output `xa` (`[1500, 1280]` FP16),
//! `O' = softmax(Q' xa^T) xa`, so no FP16 key or value is ever rounded or stored.
//!
//! `cross_absorb_f16` is the fused kernel: one block per (token, frame split) streams its
//! frames of `xa` through shared memory once, 16 frames a tile, double-buffered with
//! `cp.async`, and runs both products on each tile from shared memory with FP16 tensor
//! cores and FP32 accumulation:
//! - `S^T = xa_tile Q'^T`: frames are the MMA rows, the 20 queries (padded to 24) the
//!   columns, and the 1280-deep sum is split over the 8 warps, each holding its 160 dims of
//!   `Q'` as B fragments in registers for the whole kernel; the 8 partial tiles are summed in
//!   shared memory in warp order
//! - an online softmax in FP32 in the log2 domain (`ex2.approx`), P rounded to FP16 and the
//!   row sum taken from the unrounded P, as FlashAttention-2 does
//! - `O'^T += xa_tile^T P^T`: the 1280 output dims are the MMA rows, split over the warps
//!   (160 each, 120 FP32 accumulators a thread), the queries the columns, the frames the depth
//!
//! With one split it divides by the row sum and writes FP16 `O'`; otherwise each block writes
//! its normalized FP32 partial `[20, 1280]` and the 20 log2-sum-exps, and `cross_combine_f16`
//! merges the splits. Both stage the result in the freed `xa` tiles and store it in 16-byte
//! chunks: 120 scalar stores a thread throttled the load-store unit. Every sum runs in a fixed
//! order, so the output is the same run to run.
//!
//! The decoder runs this path for large decode buckets only (`uses_absorbed` in
//! `crates/whisperdrz-native/src/cuda/decoder_attn.rs`); prefills and small buckets attend
//! over the cached keys and values with `attn_decode_f16`

use cuda_device::float::{ex2_approx_f32, ex2_approx_ftz_f32};
use cuda_device::thread::{__unroll_config, blockIdx_x, blockIdx_y, sync_threads, threadIdx_x};
use cuda_device::vector::U32x4;
use cuda_device::{DisjointSlice, DynamicSharedArray, kernel, launch_bounds, warp};

use crate::common::f2h;
use crate::mma_f16::{FragB, SharedTile, cp_async_commit, cp_async_wait_all, cp_async16, mma};

/// Query rows per token: the decoder's heads
const QUERIES: usize = 20;
/// Encoder width, the depth of `S` and the width of `O'`
const WIDTH: usize = 1280;
/// Encoder frames
const FRAMES: u32 = 1500;
/// Frames per tile
pub const TILE: u32 = 16;
/// Tiles over the frames, the last one partly zero-filled
pub const TILES: u32 = FRAMES.div_ceil(TILE);
const WARPS: usize = 8;
const THREADS: usize = 32 * WARPS;
/// Dims of `Q'` (in `S`) and of `O'` each warp owns
const SLICE: usize = WIDTH / WARPS;
/// 16-deep steps over a warp's dims in `S`, and 16-row tiles over them in `O'`
const KS: usize = SLICE / 16;
/// 8-wide query fragments: 20 queries padded to 24
const NJ: usize = 3;
/// Shared row length of an `xa` tile in halves: 16 bytes of padding keep `ldmatrix` (plain
/// and transposed) conflict free
const LDX: usize = WIDTH + 8;
/// Floats per query row of a warp's partial `S` tile: 16 frames plus 4, which keeps the
/// accumulator stores conflict free
const LDF: usize = 20;
/// Halves per query row of the P tile: 16 frames plus 8 of padding
const LDP: usize = TILE as usize + 8;
/// P rows: 32, so two `ldmatrix.x4` cover the 3 query fragments; rows 20.. stay zero
const PROWS: usize = 32;

const XS_BYTES: usize = 2 * TILE as usize * LDX * 2;
/// Row lengths of the output staged in the `xa` tiles after the loop: FP16 `O'` rows (all 20)
/// and FP32 partial rows (10 at a time), padded so the accumulator stores spread over the banks
const LDO16: usize = WIDTH + 8;
const LDO32: usize = WIDTH + 4;
const _: () = assert!(QUERIES * LDO16 * 2 <= XS_BYTES && QUERIES / 2 * LDO32 * 4 <= XS_BYTES);
const SP_BYTES: usize = WARPS * QUERIES * LDF * 4;
const PS_BYTES: usize = PROWS * LDP * 2;
/// Dynamic shared memory the host launches `cross_absorb_f16` with: two `xa` tiles, the
/// partial `S` tiles, the P tile, and 24 each of rescale factors, row-sum inverses and
/// log2-sum-exps
pub const SMEM_BYTES: usize = XS_BYTES + SP_BYTES + PS_BYTES + 3 * 24 * 4;

const NEG_INF: f32 = f32::NEG_INFINITY;
const LOG2E: f32 = core::f32::consts::LOG2_E;

/// Starts the copy of `xa` frames `[f0, f0 + 16)` of one slot into the tile `dst`, frames at
/// or past 1500 zero-filled; one commit group
///
/// # Safety
/// Every thread of the block calls it; `src` is the 16-byte aligned start of a whole slot of
/// `xa`, and nothing reads `dst` until this thread waits on the group and a barrier follows
#[inline(always)]
unsafe fn load_tile(dst: SharedTile<LDX>, src: *const u16, f0: u32) {
    const CHUNKS: usize = TILE as usize * WIDTH / 8;
    const { assert!(CHUNKS.is_multiple_of(THREADS)) };
    let tid = threadIdx_x() as usize;
    for it in 0..CHUNKS / THREADS {
        __unroll_config::<0>();
        let c = it * THREADS + tid;
        let (r, cc) = (c / (WIDTH / 8), (c % (WIDTH / 8)) * 8);
        let frame = f0 + r as u32;
        let full = frame < FRAMES;
        // a zero-filled chunk reads nothing, but its address stays inside the slot
        let at = if full { frame as usize * WIDTH + cc } else { 0 };
        unsafe { cp_async16(dst.at(r, cc), src.wrapping_add(at), full) };
    }
    unsafe { cp_async_commit() };
}

/// Absorbed cross-attention `O' = softmax(Q' xa^T) xa` for one frame split of one token;
/// see the module docs
///
/// Grid: x = `splits`, y = `tokens` (`row * q_len + t`). Block: 256 threads, with
/// [`SMEM_BYTES`] of dynamic shared memory.
///
/// - `out`: FP16 `O'` `[tokens, 20, 1280]`, written when `splits == 1`
/// - `part_o`, `part_lse`: FP32 `[tokens, splits, 20, 1280]` and `[tokens, splits, 20]`,
///   written when `splits > 1`: each split's normalized output and log2-sum-exp per query
/// - `xa`: FP16 `[n_slots, 1500, 1280]`
/// - `q`: FP16 `Q'` `[tokens, 20, 1280]`, the softmax scale already folded in
/// - `slots`: slot per row
/// - `tiles_per_split`: 16-frame tiles per split; the host picks `splits` so that every split
///   has at least one
///
/// Traps when the buffers don't fit the grid; a row naming a slot outside `xa` writes NaN,
/// so a bad index shows in the output instead of reading another allocation
#[kernel]
#[launch_bounds(256)]
pub fn cross_absorb_f16(
    mut out: DisjointSlice<u16>,
    mut part_o: DisjointSlice<f32>,
    mut part_lse: DisjointSlice<f32>,
    xa: &[u16],
    q: &[u16],
    slots: &[i32],
    q_len: u32,
    splits: u32,
    tiles_per_split: u32,
) {
    let split = blockIdx_x();
    let token = blockIdx_y() as usize;
    let tid = threadIdx_x() as usize;
    let (w, lane) = (tid / 32, tid % 32);
    let (g, t) = (lane / 4, lane % 4);
    let row = token / q_len.max(1) as usize;
    let slot_elems = FRAMES as usize * WIDTH;
    let parts = (token + 1) * splits as usize * QUERIES;

    // everything this block touches, checked once up front: one uniform branch to one trap
    let tile_lo = split * tiles_per_split;
    let out_ok = if splits == 1 {
        out.len() >= (token + 1) * QUERIES * WIDTH
    } else {
        (part_o.len() >= parts * WIDTH) & (part_lse.len() >= parts)
    };
    let ok = out_ok
        & (q_len > 0)
        & (splits > 0)
        & (split < splits)
        & (tile_lo < TILES)
        & (row < slots.len())
        & (q.len() >= (token + 1) * QUERIES * WIDTH)
        & (xa.len() % slot_elems == 0);
    if !ok {
        panic!();
    }
    let tile_hi = (tile_lo + tiles_per_split).min(TILES);
    let frame_hi = (tile_hi * TILE).min(FRAMES);

    let smem = DynamicSharedArray::<u8, 16>::get_raw();
    let xs = [
        // SAFETY: the dynamic shared memory holds SMEM_BYTES, laid out as the consts say
        unsafe { SharedTile::<LDX>::new(smem as *mut u16) },
        unsafe { SharedTile::<LDX>::new(smem.wrapping_add(XS_BYTES / 2) as *mut u16) },
    ];
    let sp = smem.wrapping_add(XS_BYTES) as *mut f32;
    let ps_ptr = smem.wrapping_add(XS_BYTES + SP_BYTES) as *mut u16;
    // SAFETY: as above
    let ps = unsafe { SharedTile::<LDP>::new(ps_ptr) };
    let corr_s = smem.wrapping_add(XS_BYTES + SP_BYTES + PS_BYTES) as *mut f32;
    let linv_s = corr_s.wrapping_add(24);
    let lse_s = corr_s.wrapping_add(48);

    let n_slots = xa.len() / slot_elems;
    // SAFETY: row < slots.len() is part of the check above
    let slot = unsafe { *slots.get_unchecked(row) };
    if slot < 0 || slot as usize >= n_slots {
        // block-uniform, so no thread is left at a barrier
        if splits == 1 {
            let mut i = tid;
            while i < QUERIES * WIDTH {
                unsafe { *out.get_unchecked_mut(token * QUERIES * WIDTH + i) = 0x7e00 };
                i += THREADS;
            }
        } else {
            let base = (token * splits as usize + split as usize) * QUERIES;
            if tid < QUERIES {
                unsafe { *part_lse.get_unchecked_mut(base + tid) = 0.0 };
            }
            let mut i = tid;
            while i < QUERIES * WIDTH {
                unsafe { *part_o.get_unchecked_mut(base * WIDTH + i) = f32::NAN };
                i += THREADS;
            }
        }
        return;
    }
    let src = xa.as_ptr().wrapping_add(slot as usize * slot_elems);

    // SAFETY: the slot and frames lie inside xa, checked above
    unsafe { load_tile(xs[0], src, tile_lo * TILE) };

    // the P tile's padding query rows stay zero, and the padding queries' rescale factors
    // too, so their accumulator columns stay zero
    let mut i = tid;
    while i < PROWS * LDP {
        unsafe { *ps_ptr.add(i) = 0 };
        i += THREADS;
    }
    if tid < 24 {
        unsafe {
            *corr_s.add(tid) = 0.0;
            *linv_s.add(tid) = 0.0;
        }
    }

    // this warp's dims of Q' as the B operand of S^T = xa Q'^T: depth = dims, columns =
    // queries. B[k][n] = Q'[n][k], so each register is two consecutive halves of a Q' row
    let qrow = q.as_ptr().wrapping_add(token * QUERIES * WIDTH);
    let mut qf = [[FragB::ZERO; NJ]; KS];
    #[unroll]
    for kk in 0..KS {
        #[unroll]
        for j in 0..NJ {
            let n = j * 8 + g;
            if n < QUERIES {
                let k0 = w * SLICE + kk * 16 + t * 2;
                // SAFETY: n < 20 and k0 + 9 < 1280 lie inside the token's checked rows; the
                // offsets are even, so each 32-bit load is aligned
                unsafe {
                    let p = qrow.add(n * WIDTH + k0);
                    qf[kk][j] = FragB::from_bits([*(p as *const u32), *(p.add(8) as *const u32)]);
                }
            }
        }
    }

    let mut acc = [[[0.0f32; 4]; NJ]; KS];
    // online softmax state of the queries this thread owns in the softmax pass: query
    // tid / 16 and, for tid < 64, query 16 + tid / 16
    let (mut m, mut l) = ([NEG_INF; 2], [0.0f32; 2]);
    let fq = tid % 16;

    let mut tile = tile_lo;
    let mut buf = 0usize;
    while tile < tile_hi {
        // this tile has landed for every thread, and every warp is done with the last tile
        unsafe { cp_async_wait_all() };
        sync_threads();
        if tile + 1 < tile_hi {
            // SAFETY: as for the first tile; the other buffer's readers are past the barrier
            unsafe { load_tile(xs[buf ^ 1], src, (tile + 1) * TILE) };
        }
        let x = xs[buf];

        // S^T partial over this warp's dims: rows = the tile's 16 frames, columns = queries
        let mut s = [[0.0f32; 4]; NJ];
        #[unroll]
        for kk in 0..KS {
            // SAFETY: warp-uniform offsets inside the tile, after the barrier above
            let a = unsafe { x.frag_a(0, w * SLICE + kk * 16) };
            #[unroll]
            for j in 0..NJ {
                s[j] = unsafe { mma(s[j], a, qf[kk][j]) };
            }
        }
        // stored as [warp][query][frame]; padding queries are dropped
        #[unroll]
        for j in 0..NJ {
            #[unroll]
            for e in 0..4 {
                let qi = j * 8 + t * 2 + (e & 1);
                let f = g + 8 * (e / 2);
                if qi < QUERIES {
                    unsafe { *sp.add((w * QUERIES + qi) * LDF + f) = s[j][e] };
                }
            }
        }
        sync_threads();

        // softmax: 16 lanes per query, one frame each; queries 16.. in a second pass that
        // only warps 0 and 1 run, so every branch stays warp-uniform
        let f0 = tile * TILE;
        #[unroll]
        for p in 0..2 {
            let qi = p * 16 + tid / 16;
            if qi < QUERIES {
                let mut sum = 0.0f32;
                #[unroll]
                for ww in 0..WARPS {
                    sum += unsafe { *sp.add((ww * QUERIES + qi) * LDF + fq) };
                }
                // the scale is in Q'; frames past the split or past 1500 are masked
                let xl = if f0 + (fq as u32) < frame_hi {
                    sum * LOG2E
                } else {
                    NEG_INF
                };
                let mut mx = xl;
                mx = mx.max(warp::shuffle_xor_f32(mx, 1));
                mx = mx.max(warp::shuffle_xor_f32(mx, 2));
                mx = mx.max(warp::shuffle_xor_f32(mx, 4));
                mx = mx.max(warp::shuffle_xor_f32(mx, 8));
                // every tile holds a frame below frame_hi, so the new max is finite; on the
                // first tile ex2(-inf) = 0 rescales the empty state
                let m_new = m[p].max(mx);
                let corr = ex2_approx_ftz_f32(m[p] - m_new);
                let pv = ex2_approx_ftz_f32(xl - m_new);
                let mut ls = pv;
                ls += warp::shuffle_xor_f32(ls, 1);
                ls += warp::shuffle_xor_f32(ls, 2);
                ls += warp::shuffle_xor_f32(ls, 4);
                ls += warp::shuffle_xor_f32(ls, 8);
                l[p] = l[p] * corr + ls;
                m[p] = m_new;
                unsafe { *ps_ptr.add(qi * LDP + fq) = f2h(pv) };
                if fq == 0 {
                    unsafe { *corr_s.add(qi) = corr };
                }
            }
        }
        sync_threads();

        // O'^T += xa_tile^T P^T over this warp's dims: rows = dims, columns = queries
        let mut c = [[0.0f32; 2]; NJ];
        #[unroll]
        for j in 0..NJ {
            c[j] = unsafe { [*corr_s.add(j * 8 + t * 2), *corr_s.add(j * 8 + t * 2 + 1)] };
        }
        // SAFETY: warp-uniform offsets inside the P tile, after the barrier above
        let pb = unsafe { [ps.frag_b_pair_nk(0, 0), ps.frag_b_pair_nk(16, 0)] };
        #[unroll]
        for mt in 0..KS {
            // SAFETY: as above, inside the xa tile
            let a = unsafe { x.frag_a_trans(0, w * SLICE + mt * 16) };
            #[unroll]
            for j in 0..NJ {
                acc[mt][j][0] *= c[j][0];
                acc[mt][j][1] *= c[j][1];
                acc[mt][j][2] *= c[j][0];
                acc[mt][j][3] *= c[j][1];
                acc[mt][j] = unsafe { mma(acc[mt][j], a, pb[j / 2][j % 2]) };
            }
        }

        tile += 1;
        buf ^= 1;
    }

    // per query: 1 / l, and with splits the log2-sum-exp, from the lanes that own it
    #[unroll]
    for p in 0..2 {
        let qi = p * 16 + tid / 16;
        if qi < QUERIES && fq == 0 {
            // every split saw at least one frame, so l >= 1
            unsafe {
                *linv_s.add(qi) = 1.0 / l[p];
                *lse_s.add(qi) = m[p] + libm::log2f(l[p]);
            }
        }
    }
    sync_threads();

    let mut li = [[0.0f32; 2]; NJ];
    #[unroll]
    for j in 0..NJ {
        li[j] = unsafe { [*linv_s.add(j * 8 + t * 2), *linv_s.add(j * 8 + t * 2 + 1)] };
    }

    // the barrier above also put every warp past its last read of the xa tiles, which now
    // stage the result for 16-byte stores; nothing is in flight into them
    if splits == 1 {
        let st = smem as *mut u16;
        #[unroll]
        for mt in 0..KS {
            #[unroll]
            for j in 0..NJ {
                #[unroll]
                for e in 0..4 {
                    let qi = j * 8 + t * 2 + (e & 1);
                    let dim = w * SLICE + mt * 16 + g + 8 * (e / 2);
                    if qi < QUERIES {
                        let v = acc[mt][j][e] * li[j][e & 1];
                        unsafe { *st.add(qi * LDO16 + dim) = f2h(v) };
                    }
                }
            }
        }
        sync_threads();
        // the token's 20 rows are contiguous in out, inside the rows checked up front
        let dst = out.as_mut_ptr().wrapping_add(token * QUERIES * WIDTH) as *mut U32x4;
        let mut c = tid;
        while c < QUERIES * WIDTH / 8 {
            let (qi, d8) = (c / (WIDTH / 8), c % (WIDTH / 8));
            unsafe { *dst.add(c) = *(st.add(qi * LDO16 + d8 * 8) as *const U32x4) };
            c += THREADS;
        }
    } else {
        let st = smem as *mut f32;
        let base = (token * splits as usize + split as usize) * QUERIES;
        // the split's [20, 1280] partial is contiguous in part_o, inside the checked length
        let dst = part_o.as_mut_ptr().wrapping_add(base * WIDTH) as *mut U32x4;
        const HALF: usize = QUERIES / 2;
        #[unroll]
        for half in 0..2 {
            if half == 1 {
                // the first half's copy-out has read the staging rows
                sync_threads();
            }
            #[unroll]
            for mt in 0..KS {
                #[unroll]
                for j in 0..NJ {
                    #[unroll]
                    for e in 0..4 {
                        let qi = j * 8 + t * 2 + (e & 1);
                        let dim = w * SLICE + mt * 16 + g + 8 * (e / 2);
                        if qi >= half * HALF && qi < (half + 1) * HALF {
                            let v = acc[mt][j][e] * li[j][e & 1];
                            unsafe { *st.add((qi - half * HALF) * LDO32 + dim) = v };
                        }
                    }
                }
            }
            sync_threads();
            let mut c = tid;
            while c < HALF * WIDTH / 4 {
                let (r, d4) = (c / (WIDTH / 4), c % (WIDTH / 4));
                unsafe {
                    *dst.add(half * HALF * WIDTH / 4 + c) =
                        *(st.add(r * LDO32 + d4 * 4) as *const U32x4)
                };
                c += THREADS;
            }
        }
        if tid < QUERIES {
            unsafe { *part_lse.get_unchecked_mut(base + tid) = *lse_s.add(tid) };
        }
    }
}

/// Splits [`cross_combine_f16`] sums per group of loads, so one group's loads are in flight
/// together instead of each waiting on the last one's sum
const MERGE_UNROLL: usize = 8;

/// Merges the frame splits of [`cross_absorb_f16`] into FP16 `O'` `[tokens, 20, 1280]`:
/// `out = sum_s w_s O_s / sum_s w_s` with `w_s = 2^(lse_s - max lse)`, summed in split
/// order, the arithmetic `attn_combine_f16` uses for self-attention
///
/// Grid: x = `tokens * 20` (query fastest), y = 5 (256-dim chunks). Block: 64 threads, 4
/// dims each
#[kernel]
#[launch_bounds(64)]
pub fn cross_combine_f16(
    mut out: DisjointSlice<u16>,
    part_o: &[f32],
    part_lse: &[f32],
    splits: u32,
) {
    let b = blockIdx_x() as usize;
    let (token, qi) = (b / QUERIES, b % QUERIES);
    let d0 = blockIdx_y() as usize * 256 + threadIdx_x() as usize * 4;
    let n = splits as usize;
    let ok = (n > 0)
        & (d0 + 4 <= WIDTH)
        & (out.len() >= (b + 1) * WIDTH)
        & (part_lse.len() >= (token + 1) * n * QUERIES)
        & (part_o.len() >= (token + 1) * n * QUERIES * WIDTH);
    if !ok {
        panic!();
    }

    let lse = |s: usize| unsafe { *part_lse.get_unchecked((token * n + s) * QUERIES + qi) };
    let mut mx = NEG_INF;
    let mut s = 0;
    while s < n {
        mx = mx.max(lse(s));
        s += 1;
    }

    // SAFETY (loads below): split s's row of this query lies inside the checked length, and
    // d0 is a multiple of 4 in a row of 1280 floats, so each load is 16-byte aligned
    let src = part_o
        .as_ptr()
        .wrapping_add((token * n * QUERIES + qi) * WIDTH + d0);
    let (mut o, mut wsum) = ([0.0f32; 4], 0.0f32);
    let mut s0 = 0;
    while s0 < n {
        let mut v = [U32x4([0; 4]); MERGE_UNROLL];
        #[unroll]
        for k in 0..MERGE_UNROLL {
            if s0 + k < n {
                v[k] = unsafe { *(src.add((s0 + k) * QUERIES * WIDTH) as *const U32x4) };
            }
        }
        #[unroll]
        for k in 0..MERGE_UNROLL {
            if s0 + k < n {
                let l = lse(s0 + k);
                let w = if l == NEG_INF { 0.0 } else { ex2_approx_f32(l - mx) };
                wsum += w;
                #[unroll]
                for e in 0..4 {
                    o[e] += w * f32::from_bits(v[k].0[e]);
                }
            }
        }
        s0 += MERGE_UNROLL;
    }

    let fin = |x: f32| f2h(if wsum > 0.0 { x / wsum } else { f32::NAN }) as u32;
    let lo = fin(o[0]) | fin(o[1]) << 16;
    let hi = fin(o[2]) | fin(o[3]) << 16;
    // SAFETY: inside the checked row; d0 is a multiple of 4 halves, so 8-byte aligned
    unsafe { *(out.as_mut_ptr().add(b * WIDTH + d0) as *mut [u32; 2]) = [lo, hi] };
}
