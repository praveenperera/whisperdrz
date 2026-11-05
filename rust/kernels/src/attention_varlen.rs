//! Decoder attention for speculative verify steps: rows of 1 to 8 queries each
//!
//! A verify step feeds each row its last token and its drafts, so rows carry different
//! query counts. `attn_decode_f16` (`attention.rs`) takes one query count for the whole
//! batch and runs one block per query, which re-reads a row's keys and values once per
//! query: fine for self-attention's few cached keys, but cross-attention streams 7.68 MB
//! of encoder keys and values per row and layer. Here one block serves every query of a
//! (row, head, split): it streams the keys once and scores them against each query.
//!
//! Each query's arithmetic is `attn_decode_f16`'s for that query, operation for
//! operation: the same key groups and split bounds, the same shuffle tree, the same
//! online-softmax updates, skipping exactly the 64-key steps that kernel's loop never
//! reaches for the query. A batch of one-query rows therefore gives bit-identical outputs
//! and partials, which `attn_combine_f16` merges unchanged: partial `(token * HEADS + h)
//! * splits + s` for packed token `token`
//!
//! Rows come packed: row `r` owns tokens `starts[r]..starts[r + 1]` of `q`, `k_new` and
//! `v_new`, and its cache length is `lens[r]` from the host, not the device's per-slot
//! lengths, which a verify step leaves for the host to set once it knows how many drafts
//! were kept (`kv_lens_set`)

use core::ptr::addr_of_mut;

use cuda_device::thread::__unroll_config;
use cuda_device::vector::U32x4;
use cuda_device::{
    DisjointSlice, SharedArray, float, kernel, launch_bounds, launch_contract, thread, warp,
};

use crate::common::f2h;

/// Attention heads in the decoder
const HEADS: u32 = 20;
/// Dimensions per head
const HEAD_DIM: u32 = 64;
/// Model width, the length of one cached token row
const STATE: u32 = HEADS * HEAD_DIM;
/// Keys one block covers per unrolled step: 16 key groups times 4 keys each
const STEP: u32 = 64;
/// Words of shared memory per warp and query for the cross-warp merge: max, sum, 64
/// outputs
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

/// `2^x`, with `ex2.approx` as `attn_decode_f16` uses (`-inf` gives 0)
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

/// The body of the `attn_verify_q*_f16` kernels for rows of at most `Q` queries; see
/// [`attn_verify_q4_f16`] for the arguments
///
/// # Safety
/// Every thread of the block calls it, and `merge` is the calling kernel's own shared
/// array of `Q * 4 * WARP_WORDS` floats
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn verify_body<const Q: usize>(
    merge: *mut f32,
    mut out: DisjointSlice<u16>,
    mut part_o: DisjointSlice<f32>,
    mut part_lse: DisjointSlice<f32>,
    mut k_cache: DisjointSlice<u16>,
    mut v_cache: DisjointSlice<u16>,
    q: &[u16],
    k_new: &[u16],
    v_new: &[u16],
    starts: &[i32],
    lens: &[i32],
    slots: &[i32],
    q_stride: u32,
    kv_stride: u32,
    cap: u32,
    splits: u32,
    chunk: u32,
    scale_log2: f32,
) {
    let tid = thread::threadIdx_x();
    let lane = warp::lane_id();
    let wid = tid / 32;
    let sub = lane % 8;
    let group = wid * 4 + lane / 8;

    let b = thread::blockIdx_x();
    let s = thread::blockIdx_y();
    let h = b % HEADS;
    let row = b / HEADS;
    let is_self = !k_new.is_empty();

    // the row's tokens; a bad offset table fails the check below
    let first = starts.get(row as usize).copied().unwrap_or(-1);
    let end = starts.get(row as usize + 1).copied().unwrap_or(-1);
    let shape_ok = (first >= 0) & (end > first) & (end - first <= Q as i32);
    let start = first.max(0) as u32;
    // max then min, not clamp, whose min <= max assertion would compile to a trap
    let qn = (end - first).max(1).min(Q as i32) as u32;
    let last = start + qn - 1;

    // block-uniform bounds, folded into one exit: the outputs of the row's last query,
    // its query and new K/V reads, and a cache that holds whole slots
    let slot_elems = cap as usize * STATE as usize;
    let col = (h * HEAD_DIM) as usize;
    let q_end = last as usize * q_stride as usize + col + HEAD_DIM as usize;
    let kv_end = last as usize * kv_stride as usize + col + HEAD_DIM as usize;
    let last_b = (last * HEADS + h) as usize;
    let out_ok = if splits == 1 {
        out.len() >= (last as usize + 1) * STATE as usize
    } else {
        (part_o.len() >= (last_b + 1) * (splits * HEAD_DIM) as usize)
            & (part_lse.len() >= (last_b + 1) * splits as usize)
    };
    let new_ok = !is_self
        | ((k_new.len() >= kv_end) & (v_new.len() >= kv_end) & (lens.len() == slots.len()));
    let ok = shape_ok
        & out_ok
        & (row < slots.len() as u32)
        & (starts.len() == slots.len() + 1)
        & (q.len() >= q_end)
        & new_ok
        & (slot_elems > 0)
        & (k_cache.len() == v_cache.len())
        & (k_cache.len() % slot_elems.max(1) == 0)
        & (s < splits);
    if !ok {
        return;
    }

    let n_slots = k_cache.len() / slot_elems;
    // SAFETY: row < slots.len() is part of the check above
    let slot = unsafe { *slots.get_unchecked(row as usize) };
    if slot < 0 || slot as usize >= n_slots {
        // block-uniform, so no thread is left at a barrier
        if tid < HEAD_DIM {
            for t in 0..Q {
                __unroll_config::<0>();
                if (t as u32) < qn {
                    let bt = (start + t as u32) * HEADS + h;
                    if splits == 1 {
                        unsafe { *out.get_unchecked_mut((bt * HEAD_DIM + tid) as usize) = 0x7e00 };
                    } else {
                        let at = (bt * splits + s) as usize;
                        unsafe {
                            *part_o.get_unchecked_mut(at * HEAD_DIM as usize + tid as usize) =
                                f32::NAN;
                        }
                        if tid == 0 {
                            unsafe { *part_lse.get_unchecked_mut(at) = 0.0 };
                        }
                    }
                }
            }
        }
        return;
    }

    let len = if is_self {
        // SAFETY: lens.len() == slots.len() > row for self-attention, checked above
        let l = unsafe { *lens.get_unchecked(row as usize) };
        l.max(0).min(cap as i32) as u32
    } else {
        cap
    };
    // keys of the row's last query bound the block's loop; query t sees n_t of them
    let n_max = if is_self { len + qn } else { cap };
    let lo = s * chunk;
    let hi = if s + 1 == splits {
        n_max
    } else {
        (lo + chunk).min(n_max)
    };

    let cache_k = k_cache.as_mut_ptr();
    let cache_v = v_cache.as_mut_ptr();
    let slot_base = slot as usize * slot_elems + (h * HEAD_DIM + sub * 8) as usize;

    // append the row's new key t at len + t; the split-0 block of each (row, head) owns
    // those positions, 16 threads per query, and nothing in this launch reads them from
    // the cache
    if is_self && s == 0 {
        let t = tid / 16;
        let part = tid % 16;
        if t < qn && len + t < cap {
            let src = (start + t) as usize * kv_stride as usize + col + ((part % 8) * 8) as usize;
            let dst = slot as usize * slot_elems
                + ((len + t) * STATE + h * HEAD_DIM + (part % 8) * 8) as usize;
            unsafe {
                if part < 8 {
                    let v = *(k_new.as_ptr().add(src) as *const U32x4);
                    *(cache_k.add(dst) as *mut U32x4) = v;
                } else {
                    let v = *(v_new.as_ptr().add(src) as *const U32x4);
                    *(cache_v.add(dst) as *mut U32x4) = v;
                }
            }
        }
    }

    let mut qf = [[0.0f32; 8]; Q];
    for t in 0..Q {
        __unroll_config::<0>();
        if (t as u32) < qn {
            let at = (start + t as u32) as usize * q_stride as usize + col + (sub * 8) as usize;
            // SAFETY: at + 8 <= q_end, checked above
            qf[t] = widen(unsafe { *(q.as_ptr().add(at) as *const U32x4) });
        }
    }
    let new_k = k_new.as_ptr();
    let new_v = v_new.as_ptr();
    let new_base = start as usize * kv_stride as usize + (h * HEAD_DIM + sub * 8) as usize;

    let mut m = [NEG_INF; Q];
    let mut l = [0.0f32; Q];
    let mut acc = [[0.0f32; 8]; Q];

    let mut base = lo;
    while base < hi {
        let mut ks = [U32x4([0; 4]); 4];
        let mut vs = [U32x4([0; 4]); 4];
        for u in 0..4 {
            __unroll_config::<0>();
            let j = base + u as u32 * 16 + group;
            if j < hi {
                // keys below len come from the cache, the rest from this step's new rows
                // SAFETY: j < len <= cap keeps a cache read inside the slot checked above;
                // j - len < qn keeps a new-row read inside the row's k_new/v_new tokens,
                // which the up-front check covers through its last token
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

        for t in 0..Q {
            __unroll_config::<0>();
            // query t's keys and split end, as attn_decode_f16 computes them for it
            let n_t = if is_self { len + t as u32 + 1 } else { cap };
            let hi_t = if s + 1 == splits {
                n_t
            } else {
                (lo + chunk).min(n_t)
            };
            // block-uniform: the steps that kernel's loop runs for this query
            if (t as u32) < qn && base < hi_t {
                let mut x = [NEG_INF; 4];
                for u in 0..4 {
                    __unroll_config::<0>();
                    let kf = widen(ks[u]);
                    let mut d = 0.0f32;
                    for i in 0..8 {
                        __unroll_config::<0>();
                        d += qf[t][i] * kf[i];
                    }
                    // every lane shuffles, valid key or not, so the warp stays converged
                    d += warp::shuffle_xor_f32(d, 1);
                    d += warp::shuffle_xor_f32(d, 2);
                    d += warp::shuffle_xor_f32(d, 4);
                    if base + u as u32 * 16 + group < hi_t {
                        x[u] = d * scale_log2;
                    }
                }

                let m_new = m[t].max(x[0].max(x[1]).max(x[2].max(x[3])));
                if m_new != NEG_INF {
                    let corr = rescale(m[t], m_new);
                    l[t] *= corr;
                    for i in 0..8 {
                        __unroll_config::<0>();
                        acc[t][i] *= corr;
                    }
                    for u in 0..4 {
                        __unroll_config::<0>();
                        // a masked key has p = 0, and its value (loaded, finite) adds 0
                        let p = exp2(x[u] - m_new);
                        l[t] += p;
                        let vf = widen(vs[u]);
                        for i in 0..8 {
                            __unroll_config::<0>();
                            acc[t][i] += p * vf[i];
                        }
                    }
                    m[t] = m_new;
                }
            }
        }

        base += STEP;
    }

    // merge the 4 key groups of each warp (lanes 8 and 16 apart hold the same dims)
    for t in 0..Q {
        __unroll_config::<0>();
        if (t as u32) < qn {
            for k in 0..2 {
                __unroll_config::<0>();
                let off = 8u32 << k;
                let m_o = warp::shuffle_xor_f32(m[t], off);
                let l_o = warp::shuffle_xor_f32(l[t], off);
                let m_new = m[t].max(m_o);
                let a = rescale(m[t], m_new);
                let c = rescale(m_o, m_new);
                l[t] = l[t] * a + l_o * c;
                for i in 0..8 {
                    __unroll_config::<0>();
                    let o = warp::shuffle_xor_f32(acc[t][i], off);
                    acc[t][i] = acc[t][i] * a + o * c;
                }
                m[t] = m_new;
            }
        }
    }

    // then the 4 warps through shared memory, query t at words (t * 4 + warp) * WARP_WORDS
    // SAFETY: every index below is under Q * 4 * WARP_WORDS: t < Q, wid < 4,
    // sub * 8 + i < 64 and tid < 64. Raw pointers, so no thread-dependent bounds trap
    // marks the warp divergent
    if lane < 8 {
        for t in 0..Q {
            __unroll_config::<0>();
            if (t as u32) < qn {
                let w = (t * 4 + wid as usize) * WARP_WORDS;
                unsafe {
                    if lane == 0 {
                        *merge.add(w) = m[t];
                        *merge.add(w + 1) = l[t];
                    }
                    for i in 0..8 {
                        __unroll_config::<0>();
                        *merge.add(w + 2 + (sub * 8) as usize + i) = acc[t][i];
                    }
                }
            }
        }
    }
    thread::sync_threads();

    if tid < HEAD_DIM {
        for t in 0..Q {
            __unroll_config::<0>();
            if (t as u32) < qn {
                let at_t = t * 4 * WARP_WORDS;
                let mut mm = NEG_INF;
                let mut ll = 0.0f32;
                let mut o = 0.0f32;
                unsafe {
                    for w in 0..4 {
                        __unroll_config::<0>();
                        mm = mm.max(*merge.add(at_t + w * WARP_WORDS));
                    }
                    for w in 0..4 {
                        __unroll_config::<0>();
                        let c = rescale(*merge.add(at_t + w * WARP_WORDS), mm);
                        ll += *merge.add(at_t + w * WARP_WORDS + 1) * c;
                        o += *merge.add(at_t + w * WARP_WORDS + 2 + tid as usize) * c;
                    }
                }

                let bt = (start + t as u32) * HEADS + h;
                if splits == 1 {
                    // a query always has a key (its own new one, or the cross cache)
                    unsafe { *out.get_unchecked_mut((bt * HEAD_DIM + tid) as usize) = f2h(o / ll) };
                } else {
                    let at = (bt * splits + s) as usize;
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
    }
}

/// Attention for a verify step's packed rows of at most 4 queries; see the module docs
///
/// Grid: x = `rows * HEADS` with the head fastest, y = `splits`. Block: 128 threads.
///
/// - `out`: FP16 `[tokens, STATE]`, written when `splits == 1`
/// - `part_o`, `part_lse`: FP32 `[tokens * HEADS, splits, HEAD_DIM]` and `[tokens * HEADS,
///   splits]`, written when `splits > 1` (`-inf` and zeros for an empty split)
/// - `k_cache`, `v_cache`: `[n_slots, cap, HEADS, HEAD_DIM]`, appended to by
///   self-attention, only read by cross-attention
/// - `q`: token `i` at `q[i * q_stride ..][.. STATE]`
/// - `k_new`, `v_new`: the new keys and values at `[i * kv_stride ..]`, or empty for
///   cross-attention
/// - `starts`: `rows + 1` token offsets, row `r` owning tokens `starts[r]..starts[r + 1]`
/// - `lens`: each row's cache length before this step, or empty for cross-attention,
///   where every query sees all `cap` keys. Query `t` of a row attends to its `len` cached
///   keys plus new keys `0..=t`, and the row's split-0 block appends new key `t` at
///   `len + t`
/// - `slots`: cache slot per row; rows of one launch name distinct slots, except
///   one-query rows at length 0 on a shared trash slot
/// - `chunk`: keys per split; the last split takes every key past it
/// - `scale_log2`: softmax scale times `log2(e)`
///
/// A row whose slot is outside the cache gets NaN outputs; one whose offsets are bad, or
/// hold more than 4 queries, is left unwritten
#[kernel]
#[launch_bounds(128)]
pub fn attn_verify_q4_f16(
    out: DisjointSlice<u16>,
    part_o: DisjointSlice<f32>,
    part_lse: DisjointSlice<f32>,
    k_cache: DisjointSlice<u16>,
    v_cache: DisjointSlice<u16>,
    q: &[u16],
    k_new: &[u16],
    v_new: &[u16],
    starts: &[i32],
    lens: &[i32],
    slots: &[i32],
    q_stride: u32,
    kv_stride: u32,
    cap: u32,
    splits: u32,
    chunk: u32,
    scale_log2: f32,
) {
    static mut MERGE: SharedArray<f32, { 4 * 4 * WARP_WORDS }> = SharedArray::UNINIT;
    // SAFETY: MERGE is this kernel's own, sized for 4 queries
    unsafe {
        let merge = (*addr_of_mut!(MERGE)).as_mut_ptr();
        verify_body::<4>(
            merge, out, part_o, part_lse, k_cache, v_cache, q, k_new, v_new, starts, lens,
            slots, q_stride, kv_stride, cap, splits, chunk, scale_log2,
        );
    }
}

/// [`attn_verify_q4_f16`] for rows of at most 2 queries
#[kernel]
#[launch_bounds(128)]
pub fn attn_verify_q2_f16(
    out: DisjointSlice<u16>,
    part_o: DisjointSlice<f32>,
    part_lse: DisjointSlice<f32>,
    k_cache: DisjointSlice<u16>,
    v_cache: DisjointSlice<u16>,
    q: &[u16],
    k_new: &[u16],
    v_new: &[u16],
    starts: &[i32],
    lens: &[i32],
    slots: &[i32],
    q_stride: u32,
    kv_stride: u32,
    cap: u32,
    splits: u32,
    chunk: u32,
    scale_log2: f32,
) {
    static mut MERGE: SharedArray<f32, { 2 * 4 * WARP_WORDS }> = SharedArray::UNINIT;
    // SAFETY: MERGE is this kernel's own, sized for 2 queries
    unsafe {
        let merge = (*addr_of_mut!(MERGE)).as_mut_ptr();
        verify_body::<2>(
            merge, out, part_o, part_lse, k_cache, v_cache, q, k_new, v_new, starts, lens,
            slots, q_stride, kv_stride, cap, splits, chunk, scale_log2,
        );
    }
}

/// [`attn_verify_q4_f16`] for rows of at most 8 queries
#[kernel]
#[launch_bounds(128)]
pub fn attn_verify_q8_f16(
    out: DisjointSlice<u16>,
    part_o: DisjointSlice<f32>,
    part_lse: DisjointSlice<f32>,
    k_cache: DisjointSlice<u16>,
    v_cache: DisjointSlice<u16>,
    q: &[u16],
    k_new: &[u16],
    v_new: &[u16],
    starts: &[i32],
    lens: &[i32],
    slots: &[i32],
    q_stride: u32,
    kv_stride: u32,
    cap: u32,
    splits: u32,
    chunk: u32,
    scale_log2: f32,
) {
    static mut MERGE: SharedArray<f32, { 8 * 4 * WARP_WORDS }> = SharedArray::UNINIT;
    // SAFETY: MERGE is this kernel's own, sized for 8 queries
    unsafe {
        let merge = (*addr_of_mut!(MERGE)).as_mut_ptr();
        verify_body::<8>(
            merge, out, part_o, part_lse, k_cache, v_cache, q, k_new, v_new, starts, lens,
            slots, q_stride, kv_stride, cap, splits, chunk, scale_log2,
        );
    }
}

/// Sets the cache length of each row's slot to the row's position, `lens[slots[r]] =
/// positions[r]`, and the trash slot's to 0
///
/// One thread per row. A verify step leaves the per-slot lengths alone, since only the
/// host learns how many of the fed positions a row kept; the next one-token step, whose
/// positions are the kept lengths, runs this before its layers. Real rows must name
/// distinct slots
#[kernel]
pub fn kv_lens_set(mut lens: DisjointSlice<i32>, slots: &[i32], positions: &[i32], trash: u32) {
    let r = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
    if r >= slots.len() || r >= positions.len() {
        return;
    }

    let s = slots[r];
    if s < 0 || s as usize >= lens.len() {
        return;
    }

    let p = positions[r];
    unsafe { *lens.get_unchecked_mut(s as usize) = if s as u32 == trash { 0 } else { p } };
}

/// Gathers FP16 rows of 8-value groups: `out` row `i` = `src` row `idx[i]`, `row8` groups
/// per row; a negative index gives a zero row (padding) and one past `src` a NaN row.
/// One thread per group of `out`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn gather_rows_f16x8(
    mut out: DisjointSlice<U32x4>,
    src: &[U32x4],
    idx: &[i32],
    row8: u32,
) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let r = i / row8.max(1);
    let c = i - r * row8;
    let Some(mut o) = out.element_thread32(t) else {
        return;
    };

    // an Option of a 16-byte vector doesn't lower, so the reads go through indices
    // checked here
    let s = idx.get(r as usize).copied().unwrap_or(i32::MAX);
    let at = s.max(0) as usize * row8 as usize + c as usize;
    let v = if s < 0 {
        U32x4([0; 4])
    } else if s != i32::MAX && at < src.len() {
        // SAFETY: at < src.len(), checked on this branch
        unsafe { *src.get_unchecked(at) }
    } else {
        U32x4([0x7e00_7e00; 4])
    };
    o.write(v);
}
