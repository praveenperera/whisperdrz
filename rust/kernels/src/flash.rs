//! Multi-head attention on FP16, FlashAttention-2 style: bidirectional for the encoder and
//! the decoder's cross-attention, causal for the decoder's full-sequence DTW forward (the
//! decode-step attention over KV caches is in `attention`)
//!
//! The math is FlashAttention-2's forward pass, which torch's SDPA flash backend runs for
//! the reference encoder: S = Q K^T on FP16 tensor cores with FP32 accumulation, an
//! online softmax in FP32 over the raw scores with the scale folded into the exponent
//! (`exp2(s * scale * log2 e - max * scale * log2 e)`), P rounded to FP16 for P @ V, again
//! with FP32 accumulation, and the row sum taken from the unrounded P. The output is
//! divided by the row sum once at the end and rounded to FP16. The `[lq, lk]` score
//! matrix never leaves registers
//!
//! Each block owns `BM = 128` query rows of one (batch, head), eight warps of 16 rows, and
//! walks the keys in tiles of `BN = 64`. The schedule is FlashAttention-2's: the V tile's
//! copy overlaps Q K^T, and the next K tile's copy overlaps the softmax and P @ V, both
//! through `cp.async` into one K and one V buffer. Keys past `lk` in the last tile are
//! zero-filled and masked to -inf; query rows past `lq` are zero-filled and never stored.
//! The causal kernel also masks keys past each query's own index and stops at the tile
//! holding the block's last query row, as FlashAttention-2 does
//!
//! q, k, v and o are strided views: element `d` of head `h` of row `i` of batch entry `b`
//! is at `b * stride + i * ld + h * 64 + d`. Each block checks every row it will touch
//! against the slice lengths once, up front, and traps like a failed slice index if one
//! doesn't fit; the copies and stores inside then index without checks

use core::ptr::addr_of_mut;

use cuda_device::float::{ex2_approx_ftz_f32, fma_rn_f32};
use cuda_device::thread::{
    __unroll_config, blockIdx_x, blockIdx_y, blockIdx_z, sync_threads, threadIdx_x,
};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_contract, warp};

use crate::mma_f16::{
    FragA, SharedTile, cp_async_commit, cp_async_wait_all, cp_async16, mma, pack_f16,
};

/// Head dim
const D: usize = 64;
/// Shared row length in halves: 16 bytes of padding per row keep `ldmatrix` conflict free
const LD: usize = D + 8;
const WARPS: usize = 8;
const THREADS: usize = 32 * WARPS;
/// Query rows per block, 16 per warp
const BM: usize = 16 * WARPS;
/// Keys per tile
const BN: usize = 64;
/// 16-deep k-steps over the head dim in Q K^T
const KK: usize = D / 16;
/// 8-wide output fragments over the head dim
const DJ: usize = D / 8;
/// 8-wide score fragments over a key tile
const SJ: usize = BN / 8;

/// Traps when the host's geometry doesn't fit the buffers it passed, as a failed slice
/// index would
#[inline(always)]
fn require(ok: bool) {
    if !ok {
        panic!();
    }
}

/// Whether rows `[row0, row0 + rows)` of `width` elements, row `i` at `base + i * ld`, all
/// lie in a buffer of `len`. No rows always fit. Plain 64-bit math: with 32-bit strides
/// and row counts nothing wraps
#[inline(always)]
fn rows_fit(len: usize, base: usize, row0: u32, rows: u32, ld: u32, width: usize) -> bool {
    rows == 0 || base + (row0 + rows - 1) as usize * ld as usize + width <= len
}

/// Copies `ROWS` rows of 64 halves, row `r` from `src + r * ld`, into `dst`; rows at or past
/// `valid` are zero-filled. One commit group per call
///
/// # Safety
/// Every thread of the block calls it; `src` rows `[0, valid)` are readable and 16-byte
/// aligned, `valid >= 1`, and nothing reads `dst` until this thread waits on the group and
/// a barrier follows
#[inline(always)]
unsafe fn stage<const ROWS: usize>(dst: SharedTile<LD>, src: *const u16, ld: usize, valid: u32) {
    const { assert!((ROWS * D / 8).is_multiple_of(THREADS)) };
    let tid = threadIdx_x() as usize;
    for it in 0..ROWS * D / 8 / THREADS {
        // `#[unroll]` only works in `#[kernel]` bodies; this is what it expands to
        __unroll_config::<0>();
        let c = it * THREADS + tid;
        let (r, cc) = (c / (D / 8), (c % (D / 8)) * 8);
        let full = (r as u32) < valid;
        // a zero-filled chunk reads nothing, but its address stays inside the buffer
        let sr = if full { r } else { 0 };
        unsafe { cp_async16(dst.at(r, cc), src.wrapping_add(sr * ld + cc), full) };
    }
    unsafe { cp_async_commit() };
}

/// The body of both attention kernels over the tiles the kernel declared; with `CAUSAL`,
/// query `i` attends to keys `0..=i` only (`lq == lk`, checked)
///
/// # Safety
/// Every thread of the block calls it, and `qs`, `ks` and `vs` are the calling kernel's own
/// 16-byte aligned shared tiles of `BM`, `BN` and `BN` rows
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn flash_body<const CAUSAL: bool>(
    (qs, ks, vs): (SharedTile<LD>, SharedTile<LD>, SharedTile<LD>),
    mut o: DisjointSlice<u16>,
    o_ld: u32,
    o_stride: u32,
    q: &[u16],
    q_ld: u32,
    q_stride: u32,
    k: &[u16],
    k_ld: u32,
    k_stride: u32,
    v: &[u16],
    v_ld: u32,
    v_stride: u32,
    lq: u32,
    lk: u32,
    scale_log2: f32,
) {
    let (h, b) = (blockIdx_y() as usize, blockIdx_z() as usize);
    let q0 = blockIdx_x() * BM as u32;
    let tid = threadIdx_x() as usize;
    let (warp_id, lane) = (tid / 32, tid % 32);
    let (g, t, wr) = (lane / 4, lane % 4, warp_id * 16);
    let col = h * D;

    // everything this block reads or writes, checked at once up front: one uniform branch
    // to one trap, so no later check can make ptxas treat a warp as divergent
    let nq = if q0 < lq { (lq - q0).min(BM as u32) } else { 0 };
    let geometry = ((q_ld | k_ld | v_ld | o_ld | q_stride | k_stride | v_stride | o_stride) % 8
        == 0)
        & (lk >= 1)
        // causal attention is square here, where top-left and bottom-right alignment agree
        & (!CAUSAL | (lq == lk));
    require(
        geometry
            & rows_fit(q.len(), b * q_stride as usize, q0, nq, q_ld, col + D)
            & rows_fit(k.len(), b * k_stride as usize, 0, lk, k_ld, col + D)
            & rows_fit(v.len(), b * v_stride as usize, 0, lk, v_ld, col + D)
            & rows_fit(o.len(), b * o_stride as usize, q0, nq, o_ld, col + D),
    );
    if nq == 0 {
        return;
    }

    let qp = q
        .as_ptr()
        .wrapping_add(b * q_stride as usize + q0 as usize * q_ld as usize + col);
    let kp = k.as_ptr().wrapping_add(b * k_stride as usize + col);
    let vp = v.as_ptr().wrapping_add(b * v_stride as usize + col);
    let (k_ld, v_ld) = (k_ld as usize, v_ld as usize);
    // a causal block's last query row sees keys up to its own index only, so later tiles
    // would be masked out entirely
    let mut tiles = lk.div_ceil(BN as u32);
    if CAUSAL {
        tiles = tiles.min((q0 + nq).div_ceil(BN as u32));
    }

    // SAFETY: the rows were checked above; the waits and barriers below order every copy
    // before its readers and every reader before the next copy into the same buffer
    unsafe {
        stage::<BM>(qs, qp, q_ld as usize, nq);
        stage::<BN>(ks, kp, k_ld, lk.min(BN as u32));
        cp_async_wait_all();
    }
    sync_threads();

    let mut qa = [FragA::ZERO; KK];
    for kk in 0..KK {
        __unroll_config::<0>();
        // SAFETY: the whole warp runs this with warp-uniform offsets inside the tile
        qa[kk] = unsafe { qs.frag_a(wr, kk * 16) };
    }

    let mut acc = [[0.0f32; 4]; DJ];
    // running row max, already scaled to log2 units, and this thread's share of the row
    // sum, for rows g and g + 8 of the warp's 16
    let (mut m, mut l) = ([f32::NEG_INFINITY; 2], [0.0f32; 2]);

    let mut tile = 0;
    while tile < tiles {
        let k0 = tile * BN as u32;
        if tile > 0 {
            // this tile's K has landed, and every warp is done with the last tile's V
            unsafe { cp_async_wait_all() };
            sync_threads();
        }
        // SAFETY: rows [k0, k0 + valid) lie in the checked [0, lk)
        unsafe {
            stage::<BN>(
                vs,
                vp.wrapping_add(k0 as usize * v_ld),
                v_ld,
                (lk - k0).min(BN as u32),
            )
        };

        let mut s = [[0.0f32; 4]; SJ];
        for jp in 0..SJ / 2 {
            __unroll_config::<0>();
            for kk in 0..KK {
                __unroll_config::<0>();
                // SAFETY: warp-uniform offsets inside the tile, after the barrier above
                unsafe {
                    let kb = ks.frag_b_pair_nk(jp * 16, kk * 16);
                    s[2 * jp] = mma(s[2 * jp], qa[kk], kb[0]);
                    s[2 * jp + 1] = mma(s[2 * jp + 1], qa[kk], kb[1]);
                }
            }
        }

        // this tile's V has landed, and every warp is done with this tile's K
        unsafe { cp_async_wait_all() };
        sync_threads();
        if tile + 1 < tiles {
            let k1 = k0 + BN as u32;
            // SAFETY: as for V
            unsafe {
                stage::<BN>(
                    ks,
                    kp.wrapping_add(k1 as usize * k_ld),
                    k_ld,
                    (lk - k1).min(BN as u32),
                )
            };
        }

        // keys past lk in the last tile: their zero-filled scores become -inf, and with
        // CAUSAL so do keys past each query row's own index. Tile 0 holds key 0, which
        // every row may see, so every row keeps a finite max; a row whose keys in a later
        // tile are all masked gets a correction of 1 and zero weights from it
        // tested per warp (its first row is q0 + wr), so the branch stays warp-uniform
        let row0 = q0 + (wr + g) as u32;
        if (k0 + BN as u32 > lk) | (CAUSAL & (k0 + BN as u32 > q0 + wr as u32 + 1)) {
            for j in 0..SJ {
                __unroll_config::<0>();
                for e in 0..4 {
                    __unroll_config::<0>();
                    let key = k0 + (j * 8 + t * 2 + (e & 1)) as u32;
                    let row = row0 + 8 * (e / 2) as u32;
                    if (key >= lk) | (CAUSAL & (key > row)) {
                        s[j][e] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        let mut mx = [f32::NEG_INFINITY; 2];
        for j in 0..SJ {
            __unroll_config::<0>();
            mx[0] = mx[0].max(s[j][0].max(s[j][1]));
            mx[1] = mx[1].max(s[j][2].max(s[j][3]));
        }
        let mut corr = [0.0f32; 2];
        for r in 0..2 {
            __unroll_config::<0>();
            // the four lanes of a quad hold one row's 64 scores between them
            mx[r] = mx[r].max(warp::shuffle_xor_f32(mx[r], 1));
            mx[r] = mx[r].max(warp::shuffle_xor_f32(mx[r], 2));
            let scaled = m[r].max(mx[r] * scale_log2);
            // ex2(-inf) = 0 on the first tile, where acc and l are still zero
            corr[r] = ex2_approx_ftz_f32(m[r] - scaled);
            m[r] = scaled;
            l[r] *= corr[r];
        }
        for j in 0..SJ {
            __unroll_config::<0>();
            for e in 0..4 {
                __unroll_config::<0>();
                let p = ex2_approx_ftz_f32(fma_rn_f32(s[j][e], scale_log2, -m[e / 2]));
                s[j][e] = p;
                l[e / 2] += p;
            }
        }
        for dj in 0..DJ {
            __unroll_config::<0>();
            acc[dj][0] *= corr[0];
            acc[dj][1] *= corr[0];
            acc[dj][2] *= corr[1];
            acc[dj][3] *= corr[1];
        }

        for kk in 0..BN / 16 {
            __unroll_config::<0>();
            let pa = FragA::from_acc_pair(s[2 * kk], s[2 * kk + 1]);
            for djp in 0..DJ / 2 {
                __unroll_config::<0>();
                // SAFETY: warp-uniform offsets inside the tile, after the barrier above
                unsafe {
                    let vb = vs.frag_b_pair_kn(kk * 16, djp * 16);
                    acc[2 * djp] = mma(acc[2 * djp], pa, vb[0]);
                    acc[2 * djp + 1] = mma(acc[2 * djp + 1], pa, vb[1]);
                }
            }
        }
        tile += 1;
    }

    for r in 0..2 {
        __unroll_config::<0>();
        l[r] += warp::shuffle_xor_f32(l[r], 1);
        l[r] += warp::shuffle_xor_f32(l[r], 2);
        // every row saw at least one real key, so l >= 1
        l[r] = 1.0 / l[r];
    }

    // each thread stores its own two rows at columns 2t + 8dj, which no other thread of
    // any block writes, inside the rows checked up front
    let op = o
        .as_mut_ptr()
        .wrapping_add(b * o_stride as usize + q0 as usize * o_ld as usize + col);
    for r in 0..2 {
        __unroll_config::<0>();
        let row = wr + g + r * 8;
        if (row as u32) < nq {
            let out = op.wrapping_add(row * o_ld as usize);
            for dj in 0..DJ {
                __unroll_config::<0>();
                let pair = pack_f16(acc[dj][2 * r] * l[r], acc[dj][2 * r + 1] * l[r]);
                // SAFETY: row < nq and the columns lie inside the checked width; o_ld,
                // the stride and 2t + 8dj are even, so the pair is 4-byte aligned
                unsafe { *(out.add(dj * 8 + t * 2) as *mut u32) = pair };
            }
        }
    }
}

/// Attention over head dim 64: `o = softmax(q k^T * scale) v` per (batch, head), with
/// `scale_log2 = scale * log2(e)`
///
/// Grid `(ceil(lq / 128), heads, batch)`, 256 threads. Every `ld` and `stride` is a
/// multiple of 8 and every base address 16-byte aligned (the host checks), `lk >= 1`
#[kernel]
#[launch_contract(domain = 3, block = (256, 1, 1))]
pub fn flash_attn_f16_d64(
    o: DisjointSlice<u16>,
    o_ld: u32,
    o_stride: u32,
    q: &[u16],
    q_ld: u32,
    q_stride: u32,
    k: &[u16],
    k_ld: u32,
    k_stride: u32,
    v: &[u16],
    v_ld: u32,
    v_stride: u32,
    lq: u32,
    lk: u32,
    scale_log2: f32,
) {
    static mut QS: SharedArray<u16, { BM * LD }, 16> = SharedArray::UNINIT;
    static mut KS: SharedArray<u16, { BN * LD }, 16> = SharedArray::UNINIT;
    static mut VS: SharedArray<u16, { BN * LD }, 16> = SharedArray::UNINIT;
    // SAFETY: each array is this kernel's own, 16-byte aligned and sized for its rows
    let (qs, ks, vs) = unsafe {
        (
            SharedTile::<LD>::new(addr_of_mut!(QS) as *mut u16),
            SharedTile::<LD>::new(addr_of_mut!(KS) as *mut u16),
            SharedTile::<LD>::new(addr_of_mut!(VS) as *mut u16),
        )
    };

    // SAFETY: the tiles are this kernel's own, and every thread of the block runs this
    unsafe {
        flash_body::<false>(
            (qs, ks, vs),
            o, o_ld, o_stride, q, q_ld, q_stride, k, k_ld, k_stride, v, v_ld, v_stride, lq, lk,
            scale_log2,
        )
    }
}

/// Causal attention over head dim 64, as torch's SDPA flash backend runs `is_causal=True`:
/// query `i` attends to keys `0..=i`. Square only (`lq == lk`, which the decoder's DTW
/// forward over its own tokens is); otherwise as [`flash_attn_f16_d64`]
#[kernel]
#[launch_contract(domain = 3, block = (256, 1, 1))]
pub fn flash_attn_causal_f16_d64(
    o: DisjointSlice<u16>,
    o_ld: u32,
    o_stride: u32,
    q: &[u16],
    q_ld: u32,
    q_stride: u32,
    k: &[u16],
    k_ld: u32,
    k_stride: u32,
    v: &[u16],
    v_ld: u32,
    v_stride: u32,
    lq: u32,
    lk: u32,
    scale_log2: f32,
) {
    static mut QS: SharedArray<u16, { BM * LD }, 16> = SharedArray::UNINIT;
    static mut KS: SharedArray<u16, { BN * LD }, 16> = SharedArray::UNINIT;
    static mut VS: SharedArray<u16, { BN * LD }, 16> = SharedArray::UNINIT;
    // SAFETY: each array is this kernel's own, 16-byte aligned and sized for its rows
    let (qs, ks, vs) = unsafe {
        (
            SharedTile::<LD>::new(addr_of_mut!(QS) as *mut u16),
            SharedTile::<LD>::new(addr_of_mut!(KS) as *mut u16),
            SharedTile::<LD>::new(addr_of_mut!(VS) as *mut u16),
        )
    };

    // SAFETY: the tiles are this kernel's own, and every thread of the block runs this
    unsafe {
        flash_body::<true>(
            (qs, ks, vs),
            o, o_ld, o_stride, q, q_ld, q_stride, k, k_ld, k_stride, v, v_ld, v_stride, lq, lk,
            scale_log2,
        )
    }
}
