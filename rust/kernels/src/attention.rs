//! Decoder attention over the KV caches: one query row per (row, token, head), split-KV
//!
//! The decoder's self- and cross-attention both read a per-slot cache laid out token-major,
//! `[slots, cap, HEADS, HEAD_DIM]` FP16 (the layout flash-attn's `flash_attn_with_kvcache`
//! takes, and the layout the K/V projection GEMMs write). A batch row maps to its slot
//! through `slots`, and its cache length comes from `lens[slot]` on the device, so a decode
//! step needs no host read and can be captured into a CUDA graph.
//!
//! `attn_decode_f16` covers both:
//! - self-attention: `k_new`/`v_new` hold the `q_len` new keys of each row. Query `t`
//!   attends causally over the `len` cached keys plus new keys `0..=t`, read straight from
//!   `k_new`/`v_new`, and the split-0 block of query `t` appends new key `t` to the cache at
//!   `len + t`. No block reads a cache position at or past `len`, so the append and the
//!   reads of one launch never touch the same address
//! - cross-attention: `k_new` is empty, every row attends over all `cap` keys, read only
//!
//! Decode attention is bound by K/V bandwidth, so a block streams its key range with
//! 128-bit loads: 8 lanes per key (8 dims each), 4 keys per warp, 16 per block per step, 4
//! steps in flight. The host splits the key range so the grid fills the GPU at small
//! batches; with one split the kernel writes FP16 output directly, otherwise each block
//! writes its normalized partial output and log2-sum-exp for `attn_combine_f16`.
//! Softmax runs in FP32 in the log2 domain as flash-attn does (scores scaled by
//! `scale * log2(e)`, `ex2.approx`), but P stays FP32 for the P @ V product instead of
//! being rounded to FP16 for a tensor-core MMA

use cuda_device::vector::U32x4;
use cuda_device::{DisjointSlice, float, kernel, launch_bounds, thread, warp};

use crate::common::f2h;

/// Attention heads in the decoder
const HEADS: u32 = 20;
/// Dimensions per head
const HEAD_DIM: u32 = 64;
/// Model width, the length of one cached token row
const STATE: u32 = HEADS * HEAD_DIM;
/// Keys one block covers per unrolled step: 16 key groups times 4 keys each
const STEP: u32 = 64;
/// Words of shared memory per warp for the cross-warp merge: max, sum, 64 outputs
const WARP_WORDS: usize = 2 + HEAD_DIM as usize;

const NEG_INF: f32 = f32::NEG_INFINITY;

/// Widens the 8 halves of a 16-byte load to FP32 (exact)
#[inline(always)]
fn widen(v: U32x4) -> [f32; 8] {
    let w = v.0;
    [
        f32_lo(w[0]),
        f32_hi(w[0]),
        f32_lo(w[1]),
        f32_hi(w[1]),
        f32_lo(w[2]),
        f32_hi(w[2]),
        f32_lo(w[3]),
        f32_hi(w[3]),
    ]
}

#[inline(always)]
fn f32_lo(w: u32) -> f32 {
    cuda_device::convert::cvt_f32_f16x2_lo(w)
}

#[inline(always)]
fn f32_hi(w: u32) -> f32 {
    cuda_device::convert::cvt_f32_f16x2_hi(w)
}

/// `2^x`, with `ex2.approx` as flash-attn's softmax uses (`-inf` gives 0)
#[inline(always)]
fn exp2(x: f32) -> f32 {
    float::ex2_approx_f32(x)
}

/// Rescale factor from running max `old` to `new`: 0 while nothing has been seen, so an
/// all-masked state never computes `-inf - -inf`
#[inline(always)]
fn rescale(old: f32, new: f32) -> f32 {
    if old == NEG_INF { 0.0 } else { exp2(old - new) }
}

/// One query row of attention over a split of the keys; see the module docs
///
/// Grid: x = `rows * q_len * HEADS` with the head fastest, then the token, then the row
/// (so block `b` owns output columns `b * 64 ..`), y = `splits`. Block: 128 threads.
///
/// - `out`: FP16 `[rows * q_len, STATE]`, written when `splits == 1`
/// - `part_o`, `part_lse`: FP32 `[blocks_x, splits, HEAD_DIM]` and `[blocks_x, splits]`,
///   written when `splits > 1`: each split's normalized output and log2-sum-exp of its
///   scaled scores (`-inf` and zeros for an empty split)
/// - `k_cache`, `v_cache`: `[n_slots, cap, HEADS, HEAD_DIM]`, appended to by
///   self-attention, only read by cross-attention
/// - `q`: token `i = row * q_len + t` at `q[i * q_stride ..][.. STATE]`
/// - `k_new`, `v_new`: the new keys and values at `[i * kv_stride ..]`, or empty for
///   cross-attention
/// - `lens`: cache length per slot before this step (clamped to `cap`), or empty for
///   cross-attention, where every row sees all `cap` keys
/// - `slots`: cache slot per row
/// - `chunk`: keys per split; the last split takes every key past it
/// - `scale_log2`: softmax scale times `log2(e)`
///
/// A block whose row names a slot outside the cache writes NaN, so a bad index shows in
/// the output instead of reading another allocation
#[kernel]
#[launch_bounds(128)]
pub fn attn_decode_f16(
    mut out: DisjointSlice<u16>,
    mut part_o: DisjointSlice<f32>,
    mut part_lse: DisjointSlice<f32>,
    mut k_cache: DisjointSlice<u16>,
    mut v_cache: DisjointSlice<u16>,
    q: &[u16],
    k_new: &[u16],
    v_new: &[u16],
    lens: &[i32],
    slots: &[i32],
    q_len: u32,
    q_stride: u32,
    kv_stride: u32,
    cap: u32,
    splits: u32,
    chunk: u32,
    scale_log2: f32,
) {
    static mut MERGE: cuda_device::SharedArray<f32, { 4 * WARP_WORDS }> =
        cuda_device::SharedArray::UNINIT;

    let tid = thread::threadIdx_x();
    let lane = warp::lane_id();
    let wid = tid / 32;
    let sub = lane % 8;
    let group = wid * 4 + lane / 8;

    let b = thread::blockIdx_x();
    let s = thread::blockIdx_y();
    let h = b % HEADS;
    let token = b / HEADS;
    // q_len == 0 fails the check below; max(1) keeps the division free of a trap
    let row = token / q_len.max(1);
    let t = token % q_len.max(1);
    let is_self = !k_new.is_empty();

    // block-uniform bounds, folded into one exit: the output, the query and the new K/V
    // reads, and a cache that holds whole slots
    let slot_elems = cap as usize * STATE as usize;
    let q_at = token as usize * q_stride as usize + (h * HEAD_DIM) as usize;
    let kv_at = token as usize * kv_stride as usize + (h * HEAD_DIM) as usize;
    let out_ok = if splits == 1 {
        out.len() >= (token as usize + 1) * STATE as usize
    } else {
        (part_o.len() >= (b as usize + 1) * (splits * HEAD_DIM) as usize)
            & (part_lse.len() >= (b as usize + 1) * splits as usize)
    };
    let new_ok = !is_self
        | ((k_new.len() >= kv_at + HEAD_DIM as usize) & (v_new.len() >= kv_at + HEAD_DIM as usize));
    let ok = out_ok
        & (row < slots.len() as u32)
        & (q.len() >= q_at + HEAD_DIM as usize)
        & new_ok
        & (slot_elems > 0)
        & (k_cache.len() == v_cache.len())
        & (k_cache.len() % slot_elems.max(1) == 0)
        & (s < splits)
        & (q_len > 0);
    if !ok {
        return;
    }

    let n_slots = k_cache.len() / slot_elems;
    // SAFETY: row < slots.len() is part of the check above
    let slot = unsafe { *slots.get_unchecked(row as usize) };
    if slot < 0 || slot as usize >= n_slots {
        // block-uniform, so no thread is left at a barrier
        if tid < HEAD_DIM {
            if splits == 1 {
                unsafe { *out.get_unchecked_mut((b * HEAD_DIM + tid) as usize) = 0x7e00 };
            } else {
                let at = (b * splits + s) as usize;
                unsafe {
                    *part_o.get_unchecked_mut(at * HEAD_DIM as usize + tid as usize) = f32::NAN;
                }
                if tid == 0 {
                    unsafe { *part_lse.get_unchecked_mut(at) = 0.0 };
                }
            }
        }
        return;
    }

    let len = if is_self {
        let l = lens.get(slot as usize).copied().unwrap_or(0);
        // max then min, not clamp, whose min <= max assertion would compile to a trap
        l.max(0).min(cap as i32) as u32
    } else {
        cap
    };
    let n_keys = if is_self { len + t + 1 } else { cap };
    let lo = s * chunk;
    let hi = if s + 1 == splits { n_keys } else { (lo + chunk).min(n_keys) };

    let cache_k = k_cache.as_mut_ptr();
    let cache_v = v_cache.as_mut_ptr();
    let slot_base = slot as usize * slot_elems + (h * HEAD_DIM + sub * 8) as usize;

    // append new key t of this row at len + t; the split-0 block of each (row, t, head)
    // owns that position, and nothing in this launch reads it from the cache
    if is_self && s == 0 && tid < 16 && len + t < cap {
        let src = kv_at + ((tid % 8) * 8) as usize;
        let dst = slot as usize * slot_elems
            + ((len + t) * STATE + h * HEAD_DIM + (tid % 8) * 8) as usize;
        unsafe {
            if tid < 8 {
                let v = *(k_new.as_ptr().add(src) as *const U32x4);
                *(cache_k.add(dst) as *mut U32x4) = v;
            } else {
                let v = *(v_new.as_ptr().add(src) as *const U32x4);
                *(cache_v.add(dst) as *mut U32x4) = v;
            }
        }
    }

    let qf = widen(unsafe { *(q.as_ptr().add(q_at + (sub * 8) as usize) as *const U32x4) });
    let new_k = k_new.as_ptr();
    let new_v = v_new.as_ptr();
    let new_base = (row * q_len) as usize * kv_stride as usize + (h * HEAD_DIM + sub * 8) as usize;

    let mut m = NEG_INF;
    let mut l = 0.0f32;
    let mut acc = [0.0f32; 8];

    let mut base = lo;
    while base < hi {
        let mut ks = [U32x4([0; 4]); 4];
        let mut vs = [U32x4([0; 4]); 4];
        #[unroll]
        for u in 0..4 {
            let j = base + u as u32 * 16 + group;
            if j < hi {
                // keys below len come from the cache, the rest from this step's new rows
                // SAFETY: j < len <= cap keeps a cache read inside the slot checked above;
                // j - len <= t keeps a new-row read inside the k_new/v_new rows of this
                // row, which the up-front check covers through token = row * q_len + t
                unsafe {
                    let (kp, vp) = if j < len {
                        let at = slot_base + (j * STATE) as usize;
                        (cache_k.add(at) as *const u16, cache_v.add(at) as *const u16)
                    } else {
                        let at = new_base + (j - len) as usize * kv_stride as usize;
                        (new_k.add(at), new_v.add(at))
                    };
                    ks[u] = *(kp as *const U32x4);
                    vs[u] = *(vp as *const U32x4);
                }
            }
        }

        let mut x = [NEG_INF; 4];
        #[unroll]
        for u in 0..4 {
            let kf = widen(ks[u]);
            let mut d = 0.0f32;
            #[unroll]
            for i in 0..8 {
                d += qf[i] * kf[i];
            }
            // every lane shuffles, valid key or not, so the warp stays converged
            d += warp::shuffle_xor_f32(d, 1);
            d += warp::shuffle_xor_f32(d, 2);
            d += warp::shuffle_xor_f32(d, 4);
            if base + u as u32 * 16 + group < hi {
                x[u] = d * scale_log2;
            }
        }

        let m_new = m.max(x[0].max(x[1]).max(x[2].max(x[3])));
        if m_new != NEG_INF {
            let corr = rescale(m, m_new);
            l *= corr;
            #[unroll]
            for i in 0..8 {
                acc[i] *= corr;
            }
            #[unroll]
            for u in 0..4 {
                let p = exp2(x[u] - m_new);
                l += p;
                let vf = widen(vs[u]);
                #[unroll]
                for i in 0..8 {
                    acc[i] += p * vf[i];
                }
            }
            m = m_new;
        }

        base += STEP;
    }

    // merge the 4 key groups of each warp (lanes 8 and 16 apart hold the same dims)
    #[unroll]
    for k in 0..2 {
        let off = 8u32 << k;
        let m_o = warp::shuffle_xor_f32(m, off);
        let l_o = warp::shuffle_xor_f32(l, off);
        let m_new = m.max(m_o);
        let a = rescale(m, m_new);
        let c = rescale(m_o, m_new);
        l = l * a + l_o * c;
        #[unroll]
        for i in 0..8 {
            let o = warp::shuffle_xor_f32(acc[i], off);
            acc[i] = acc[i] * a + o * c;
        }
        m = m_new;
    }

    // then the 4 warps through shared memory
    // SAFETY: every index below is under 4 * WARP_WORDS: wid < 4, sub * 8 + i < 64 and
    // tid < 64. Raw pointers, so no thread-dependent bounds trap marks the warp divergent
    let merge = unsafe { (*core::ptr::addr_of_mut!(MERGE)).as_mut_ptr() };
    if lane < 8 {
        let w = wid as usize * WARP_WORDS;
        unsafe {
            if lane == 0 {
                *merge.add(w) = m;
                *merge.add(w + 1) = l;
            }
            #[unroll]
            for i in 0..8 {
                *merge.add(w + 2 + (sub * 8) as usize + i) = acc[i];
            }
        }
    }
    thread::sync_threads();

    if tid < HEAD_DIM {
        let mut mm = NEG_INF;
        let mut ll = 0.0f32;
        let mut o = 0.0f32;
        unsafe {
            #[unroll]
            for w in 0..4 {
                mm = mm.max(*merge.add(w * WARP_WORDS));
            }
            #[unroll]
            for w in 0..4 {
                let c = rescale(*merge.add(w * WARP_WORDS), mm);
                ll += *merge.add(w * WARP_WORDS + 1) * c;
                o += *merge.add(w * WARP_WORDS + 2 + tid as usize) * c;
            }
        }

        if splits == 1 {
            // a row always has a key (its own new one, or the cross cache), so ll > 0
            unsafe { *out.get_unchecked_mut((b * HEAD_DIM + tid) as usize) = f2h(o / ll) };
        } else {
            let at = (b * splits + s) as usize;
            let (o, lse) = if ll > 0.0 {
                (o / ll, mm + libm::log2f(ll))
            } else {
                (0.0, NEG_INF)
            };
            unsafe {
                *part_o.get_unchecked_mut(at * HEAD_DIM as usize + tid as usize) = o;
            }
            if tid == 0 {
                unsafe { *part_lse.get_unchecked_mut(at) = lse };
            }
        }
    }
}

/// Splits [`attn_combine_f16`] loads per round trip
const COMBINE_BATCH: usize = 16;

/// Merges the split partials of `attn_decode_f16` into FP16 output
///
/// Grid: x = blocks of the attention launch, one per (row, token, head). Block: 64 threads,
/// one per output dimension. `out[b * 64 + d] = sum_s w_s o_s / sum_s w_s` with
/// `w_s = 2^(lse_s - max lse)`
#[kernel]
#[launch_bounds(64)]
pub fn attn_combine_f16(
    mut out: DisjointSlice<u16>,
    part_o: &[f32],
    part_lse: &[f32],
    splits: u32,
) {
    let b = thread::blockIdx_x() as usize;
    let d = thread::threadIdx_x() as usize;
    let n = splits as usize;
    let ok = (d < HEAD_DIM as usize)
        & (n > 0)
        & (out.len() >= (b + 1) * HEAD_DIM as usize)
        & (part_lse.len() >= (b + 1) * n)
        & (part_o.len() >= (b + 1) * n * HEAD_DIM as usize);
    if !ok {
        return;
    }

    // the loads go out COMBINE_BATCH at a time and the sums take them in split order, so
    // the result is the plain loop's while the latency is one round trip per batch, not
    // one per split (24 splits at one cross-attention row)
    let lse = &part_lse[b * n..(b + 1) * n];
    let mut mx = NEG_INF;
    let mut i = 0;
    while i < n {
        let mut v = [NEG_INF; COMBINE_BATCH];
        #[unroll]
        for j in 0..COMBINE_BATCH {
            if i + j < n {
                v[j] = lse[i + j];
            }
        }
        #[unroll]
        for j in 0..COMBINE_BATCH {
            mx = mx.max(v[j]);
        }
        i += COMBINE_BATCH;
    }

    let mut wsum = 0.0f32;
    let mut o = 0.0f32;
    let mut i = 0;
    while i < n {
        let mut l = [NEG_INF; COMBINE_BATCH];
        let mut p = [0.0f32; COMBINE_BATCH];
        #[unroll]
        for j in 0..COMBINE_BATCH {
            if i + j < n {
                l[j] = lse[i + j];
                p[j] = part_o[(b * n + i + j) * HEAD_DIM as usize + d];
            }
        }
        #[unroll]
        for j in 0..COMBINE_BATCH {
            if i + j < n {
                let w = rescale(l[j], mx);
                wsum += w;
                o += w * p[j];
            }
        }
        i += COMBINE_BATCH;
    }

    let v = if wsum > 0.0 { o / wsum } else { f32::NAN };
    unsafe { *out.get_unchecked_mut(b * HEAD_DIM as usize + d) = f2h(v) };
}

/// Updates the per-slot cache lengths for the rows of a step
///
/// One thread per row: `lens[slot] = keep * lens[slot] + add`, except the trash slot that
/// padding rows share, which goes back to 0 (several padding rows may write it at once,
/// all the same value). Real rows must name distinct slots. Prefill runs it with
/// `keep = 0, add = 0` before the layers and `keep = 1, add = q_len` after them; a decode
/// step runs only the second
#[kernel]
pub fn kv_lens(mut lens: DisjointSlice<i32>, slots: &[i32], keep: u32, add: u32, trash: u32) {
    let r = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
    if r >= slots.len() {
        return;
    }

    let s = slots[r];
    if s < 0 || s as usize >= lens.len() {
        return;
    }

    unsafe {
        let p = lens.get_unchecked_mut(s as usize);
        *p = if s as u32 == trash {
            0
        } else {
            *p * keep as i32 + add as i32
        };
    }
}
