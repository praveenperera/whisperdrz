//! The text decoder's own kernels: the token and positional embedding gather, and the
//! transposed copy of the token embedding the LM head multiplies by

use cuda_device::{DisjointSlice, kernel, launch_contract, thread};

use crate::common::{f2h, h2f};
use crate::pointwise::H8;

/// The decoder's input embedding, eight values per thread:
/// `out[r] = f16(tok_emb[tokens[r]] + pos_emb[positions[r]])`, as `TextDecoder` sums its
/// two FP16 embeddings
///
/// With a non-empty `timing` (`[n_timing, cols8]`, the DTW forward's
/// `_get_timing_embedding`), the sum is rounded once more after adding
/// `f16(timing[clamp(tok - ts_begin, 0, n_timing - 1)] * is_timing)`, the FP16 product
/// torch forms, where `is_timing` is 1 for `ts_begin <= tok < ts_begin + n_timing` and 0
/// otherwise. `out` is `[rows, cols8]` in 8-value chunks; `tokens` and `positions` hold
/// one `i32` per row, read on the device so a captured step takes whatever the host last
/// wrote. A token or position outside its table traps as a failed slice index
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn embed_f16x8(
    mut out: DisjointSlice<H8>,
    tok_emb: &[H8],
    pos_emb: &[H8],
    timing: &[H8],
    tokens: &[i32],
    positions: &[i32],
    cols8: u32,
    ts_begin: u32,
    n_timing: u32,
) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let Some(mut out) = out.element_thread32(t) else {
        return;
    };
    let (row, c) = ((i / cols8) as usize, (i % cols8) as usize);
    let (tok, pos) = (tokens[row], positions[row]);
    // a negative id wraps far past the table, so the checked index traps on it
    let (e, p) = (
        tok_emb[tok as u32 as usize * cols8 as usize + c].unpack(),
        pos_emb[pos as u32 as usize * cols8 as usize + c].unpack(),
    );
    let mut x = [0.0f32; 8];
    #[unroll]
    for k in 0..8 {
        x[k] = e[k] + p[k];
    }

    if !timing.is_empty() {
        let rel = tok - ts_begin as i32;
        let on = (rel >= 0) & (rel < n_timing as i32);
        let ti = rel.clamp(0, n_timing as i32 - 1) as usize;
        let te = timing[ti * cols8 as usize + c].unpack();
        let mask = if on { 1.0f32 } else { 0.0 };
        #[unroll]
        for k in 0..8 {
            // torch rounds the sum, then the masked timing product, before adding them
            x[k] = h2f(f2h(x[k])) + h2f(f2h(te[k] * mask));
        }
    }
    out.write(H8::pack(x));
}

/// `out[c, r] = x[r, c]` for FP16 `x` `[rows, cols]`, one output element per thread: the
/// LM head's contiguous transposed token embedding, built once at load time
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn transpose_f16(mut out: DisjointSlice<u16>, x: &[u16], rows: u32, cols: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let Some(mut out) = out.element_thread32(t) else {
        return;
    };
    let (c, r) = (i / rows, i % rows);
    out.write(x[r as usize * cols as usize + c as usize]);
}
