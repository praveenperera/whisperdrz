//! The native text decoder against f64 and torch
//!
//! `decoder_kernels_match_f64` checks the decoder's own kernels (causal attention, the
//! embedding gather, the transposed copy, the DTW capture and prep kernels) element by
//! element on exact inputs, against bounds derived from each kernel's own roundings, with
//! NaN-filled outputs. `decoder_layers_match_torch` runs the traced short3 prefill and two
//! decode steps (q_len 1, slots out of order, one padding row on the trash slot, ragged
//! cache lengths) and judges ours and torch FP16 against an FP32 truth on the same FP16
//! weights (`rust/tools/decoder_layers.py`): isolated (our layer N on torch FP16's input
//! and caches) and chained (ours end to end from the tokens), on real rows only (padding
//! rows must stay finite and can't change real rows, checked with a poisoned trash slot).
//! These batches are far below `ABSORBED_MIN_ROWS`, so their cross-attention runs over the
//! K/V caches; `tests/gpu_decoder_attn.rs` checks the absorbed path of large buckets.
//! Each sub-op of each block (the three LayerNorms, self-attention, cross-attention, the
//! MLP), run on torch FP16's input to it, must have max-abs and mean-abs errors at most
//! 1.5x torch FP16's. Whole layers, isolated and chained, assert mean-abs at 1.5x and
//! print max-abs against torch FP16 (chained: the worst of torch FP16, packed QKV, the SDPA
//! fallback and memory-efficient SDPA, each exact in real arithmetic) without asserting
//! it: with every sub-op inside the bar, an FP16 ulp flip in one sub-op's output still
//! moves the row's LayerNorm statistics and lands in single elements of the block output,
//! which reached 1.5-1.9x on three real-row elements of the short3 trace, while torch's
//! own variants reach 1.86x chained. The end-to-end gate (`rust/gate/gate.py`) judges
//! what that drift does to the transcripts. Skips unless `WHISPERDRZ_GPU=1` (and for the torch check
//! `WHISPERDRZ_WEIGHTS` and `WHISPERDRZ_DECODER_REF`); run through the GPU queue with
//! `rust/scripts/gpu-test.sh`
#![cfg(feature = "cuda")]

#[allow(dead_code)]
#[path = "support/numerics.rs"]
mod numerics;
#[allow(dead_code)]
#[path = "support/parity.rs"]
mod parity;

use std::path::PathBuf;

use eyre::{Result, bail, ensure, eyre};
use half::f16;
use numerics::{Err, NAN16, Rng, f16_to_f64, gpu_enabled, ulp16};
use parity::{Check, MAX_RATIO, Row, bits_f64, download16, f16_bits, f64s, report};
use serde_json::Value;
use whisperdrz_fixtures::{ArrayData, TraceSet};
use whisperdrz_native::cuda::decoder::{
    self, D_MODEL, MAX_ROWS, N_LAYER, N_VOCAB, PREFILL_LEN, SubOp, Workspace,
};
use whisperdrz_native::cuda::decoder_attn::{Batch, CROSS_LEN, SELF_CAP};
use whisperdrz_native::cuda::ops::{self, HeadView};
use whisperdrz_native::cuda::{
    Arg, Blas, Buf, Decoder, DecoderAttn, Gpu, KvCaches, Weights, blocks,
};

const U32: f64 = 1.0 / (1u64 << 24) as f64;

fn to_f64(bits: &[u16]) -> Vec<f64> {
    bits.iter().copied().map(f16_to_f64).collect()
}

fn f16_bits_of(x: f32) -> u16 {
    f16::from_f32(x).to_bits()
}

fn h(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

fn nan_buf(gpu: &Gpu, n: usize) -> Result<Buf> {
    gpu.upload(vec![NAN16; n])
}

fn nan_f32(gpu: &Gpu, n: usize) -> Result<Buf> {
    gpu.upload(vec![f32::NAN; n])
}

/// One causal attention case: q, k, v packed in `[batch, len, 3 * heads * 64]` as the DTW
/// forward's QKV GEMM writes them
struct CausalCase {
    name: &'static str,
    batch: usize,
    heads: usize,
    len: usize,
    q_std: f64,
}

/// Checks every output element against the exact f64 causal attention with the bound
/// `0.5 ulp16(o) + (2^-11 + 1e-4) * sum_j p_j |v_j|` of `gpu_ops`'s attention check: FP16
/// rounding of the output and of each softmax weight in P @ V, and 1e-4 relative for the
/// FP32 accumulation, `ex2.approx` and the FP32 scores
fn causal_case(gpu: &Gpu, c: &CausalCase, seed: u64) -> Result<()> {
    let mut rng = Rng::new(seed);
    let (hd, scale) = (c.heads * 64, 0.125f64);
    let ld = 3 * hd;
    let buf: Vec<u16> = rng
        .f16_normal(c.batch * c.len * ld, 0.0, 1.0)
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if i % ld < hd {
                f16::from_f64(f16_to_f64(v) * c.q_std).to_bits()
            } else {
                v
            }
        })
        .collect();
    let o_ld = hd + 8;
    let qd = gpu.upload(buf.clone())?;
    let od = nan_buf(gpu, c.batch * c.len * o_ld)?;
    let view = |ptr: u64, ld: usize, off: usize| HeadView {
        ptr: ptr + 2 * off as u64,
        len: c.batch * c.len * ld - off,
        ld,
        batch_stride: c.len * ld,
    };
    ops::attention_causal(
        gpu,
        view(od.ptr(), o_ld, 0),
        (
            view(qd.ptr(), ld, 0),
            view(qd.ptr(), ld, hd),
            view(qd.ptr(), ld, 2 * hd),
        ),
        (c.batch, c.heads, c.len),
        scale as f32,
    )?;
    let o: Vec<u16> = gpu.download(&od)?;

    let x = to_f64(&buf);
    let (mut got, mut want, mut worst) = (Vec::new(), Vec::new(), 0f64);
    let mut s = vec![0f64; c.len];
    for b in 0..c.batch {
        for hh in 0..c.heads {
            for i in 0..c.len {
                let qrow = &x[(b * c.len + i) * ld + hh * 64..][..64];
                // query i sees keys 0..=i only
                let keys = i + 1;
                for (j, sj) in s[..keys].iter_mut().enumerate() {
                    let krow = &x[(b * c.len + j) * ld + hd + hh * 64..][..64];
                    *sj = scale * qrow.iter().zip(krow).map(|(a, b)| a * b).sum::<f64>();
                }
                let m = s[..keys].iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let l: f64 = s[..keys].iter().map(|v| (v - m).exp()).sum();
                for d in 0..64 {
                    let (mut acc, mut mag) = (0f64, 0f64);
                    for (j, sj) in s[..keys].iter().enumerate() {
                        let p = (sj - m).exp() / l;
                        let vj = x[(b * c.len + j) * ld + 2 * hd + hh * 64 + d];
                        acc += p * vj;
                        mag += p * vj.abs();
                    }
                    let g = f16_to_f64(o[(b * c.len + i) * o_ld + hh * 64 + d]);
                    let bound = 0.5 * ulp16(acc) + (2f64.powi(-11) + 1e-4) * mag;
                    ensure!(
                        (g - acc).abs() <= bound,
                        "causal {}: o[{b}, {i}, {hh}, {d}] = {g}, f64 {acc}, bound {bound:.3e}",
                        c.name
                    );
                    worst = worst.max((g - acc).abs() / bound);
                    got.push(g);
                    want.push(acc);
                }
            }
        }
    }

    for row in o.chunks(o_ld) {
        ensure!(
            row[hd..].iter().all(|&v| v == NAN16),
            "causal {}: wrote into o's row padding",
            c.name
        );
    }
    let err = Err::of(&got, &want)?;
    println!(
        "causal attention {}: B={} H={} T={}: {} values, max {:.3e} mean {:.3e}, worst error/bound {worst:.3}",
        c.name,
        c.batch,
        c.heads,
        c.len,
        got.len(),
        err.max_abs,
        err.mean_abs
    );
    Ok(())
}

/// `embed_f16x8` on small random tables, with and without the timing table, against
/// torch's roundings: `f16(e + p)` from FP32 sums of the FP16 values, then for timing
/// `f16(f16(e + p) + f16(te * is_timing))`. Every value is checked bit for bit
fn embed_case(gpu: &Gpu, timing: bool) -> Result<()> {
    let (vocab, ctx, n_timing, ts_begin, cols) = (300usize, 40usize, 50usize, 200u32, 64usize);
    let cols8 = cols / 8;
    let mut rng = Rng::new(if timing { 71 } else { 70 });
    let tok = rng.f16_normal(vocab * cols, 0.0, 0.5);
    let pos = rng.f16_normal(ctx * cols, 0.0, 0.5);
    let te = rng.f16_normal(n_timing * cols, 0.0, 1.0);
    // the edges of the timing range, both sides, plus a spread over the whole table
    let mut tokens: Vec<i32> = vec![0, 199, 200, 201, 248, 249, 250, 299];
    tokens.extend((0..56).map(|i| (i * 37 % vocab) as i32));
    let rows = tokens.len();
    let positions: Vec<i32> = (0..rows).map(|i| (i * 7 % ctx) as i32).collect();

    let (tok_d, pos_d, te_d) = (
        gpu.upload(tok.clone())?,
        gpu.upload(pos.clone())?,
        gpu.upload(te.clone())?,
    );
    let (tokens_d, positions_d) = (gpu.upload(tokens.clone())?, gpu.upload(positions.clone())?);
    let out = nan_buf(gpu, rows * cols)?;
    gpu.launch(
        "embed_f16x8",
        (blocks(rows * cols8, 256), 1, 1),
        (256, 1, 1),
        0,
        &[
            Arg::slice(out.ptr(), rows * cols8),
            Arg::slice(tok_d.ptr(), vocab * cols8),
            Arg::slice(pos_d.ptr(), ctx * cols8),
            if timing {
                Arg::slice(te_d.ptr(), n_timing * cols8)
            } else {
                Arg::EMPTY
            },
            Arg::slice(tokens_d.ptr(), rows),
            Arg::slice(positions_d.ptr(), rows),
            cols8.into(),
            ts_begin.into(),
            n_timing.into(),
        ],
    )?;
    let got: Vec<u16> = gpu.download(&out)?;

    let mut on = 0;
    for (r, (&t, &p)) in tokens.iter().zip(&positions).enumerate() {
        let rel = t - ts_begin as i32;
        let is_timing = (0..n_timing as i32).contains(&rel);
        on += usize::from(is_timing);
        let ti = rel.clamp(0, n_timing as i32 - 1) as usize;
        for c in 0..cols {
            let e = h(tok[t as usize * cols + c]) + h(pos[p as usize * cols + c]);
            let mut want = f16_bits_of(e);
            if timing {
                let mask = if is_timing { 1.0f32 } else { 0.0 };
                let add = h(f16_bits_of(h(te[ti * cols + c]) * mask));
                want = f16_bits_of(h(want) + add);
            }
            ensure!(
                got[r * cols + c] == want,
                "embed (timing {timing}) row {r} token {t} col {c}: {:#06x}, want {want:#06x}",
                got[r * cols + c]
            );
        }
    }
    println!(
        "embed_f16x8 timing={timing}: {} values bit-exact ({on} timing rows)",
        rows * cols
    );
    Ok(())
}

fn transpose_case(gpu: &Gpu) -> Result<()> {
    let (rows, cols) = (37usize, 24usize);
    let x = Rng::new(72).f16_normal(rows * cols, 0.0, 1.0);
    let xd = gpu.upload(x.clone())?;
    let out = nan_buf(gpu, rows * cols)?;
    gpu.launch(
        "transpose_f16",
        (blocks(rows * cols, 256), 1, 1),
        (256, 1, 1),
        0,
        &[
            Arg::slice(out.ptr(), rows * cols),
            Arg::slice(xd.ptr(), rows * cols),
            rows.into(),
            cols.into(),
        ],
    )?;
    let got: Vec<u16> = gpu.download(&out)?;
    for r in 0..rows {
        for c in 0..cols {
            ensure!(got[c * rows + r] == x[r * cols + c], "transpose ({r}, {c})");
        }
    }
    println!("transpose_f16: {} values bit-exact", rows * cols);
    Ok(())
}

/// `dtw_gather_head_f32`: one head of FP16 rows widened and scaled, one FP32 product each,
/// so bit-exact against `f32(x) * scale`
fn gather_case(gpu: &Gpu) -> Result<()> {
    let (rows, ld, col) = (10usize, 128usize, 64usize);
    let scale = 64f64.powf(-0.25) as f32;
    let src = Rng::new(73).f16_normal(rows * ld, 0.0, 2.0);
    let sd = gpu.upload(src.clone())?;
    let out = nan_f32(gpu, rows * 64)?;
    gpu.launch(
        "dtw_gather_head_f32",
        (blocks(rows * 64, 256), 1, 1),
        (256, 1, 1),
        0,
        &[
            Arg::slice(out.ptr(), rows * 64),
            Arg::slice(sd.ptr(), rows * ld),
            ld.into(),
            col.into(),
            scale.into(),
        ],
    )?;
    let got: Vec<f32> = gpu.download(&out)?;
    for r in 0..rows {
        for d in 0..64 {
            let want = h(src[r * ld + col + d]) * scale;
            ensure!(
                got[r * 64 + d].to_bits() == want.to_bits(),
                "gather ({r}, {d}): {}, want {want}",
                got[r * 64 + d]
            );
        }
    }
    println!("dtw_gather_head_f32: {} values bit-exact", rows * 64);
    Ok(())
}

/// `dtw_softmax_f32` on `[heads, batch, t, 1500]` scores with per-row audio caps, against
/// the f64 softmax over the columns under the cap. Masked columns must be exactly 0; the
/// rest stay within `(|s - max| + 24) * 2^-24` relative: the FP32 subtraction feeds
/// `|s - max| * 2^-24` into the exponent, `expf` is within 2 ulp, and the sum's at most 19
/// additions of positive values (6 per thread, 5 shuffle levels, 8 warps) and the division
/// add one rounding each
fn softmax_case(gpu: &Gpu) -> Result<()> {
    let (heads, batch, t, cols) = (2usize, 2usize, 3usize, 1500usize);
    let caps = [37i32, 1500];
    let mut rng = Rng::new(74);
    let w: Vec<f32> = (0..heads * batch * t * cols)
        .map(|_| (5.0 * rng.normal()) as f32)
        .collect();
    let rows = heads * batch * t;
    let wd = gpu.upload(w.clone())?;
    let capd = gpu.upload(caps.to_vec())?;
    gpu.launch(
        "dtw_softmax_f32",
        (rows as u32, 1, 1),
        (256, 1, 1),
        0,
        &[
            Arg::slice(wd.ptr(), rows * cols),
            Arg::slice(capd.ptr(), batch),
            rows.into(),
            t.into(),
            batch.into(),
            cols.into(),
        ],
    )?;
    let got: Vec<f32> = gpu.download(&wd)?;

    let (mut worst, mut checked) = (0f64, 0usize);
    for row in 0..rows {
        let cap = caps[(row / t) % batch] as usize;
        let s = &w[row * cols..][..cols];
        let g = &got[row * cols..][..cols];
        let m = s[..cap]
            .iter()
            .map(|&v| f64::from(v))
            .fold(f64::NEG_INFINITY, f64::max);
        let l: f64 = s[..cap].iter().map(|&v| (f64::from(v) - m).exp()).sum();
        for a in 0..cols {
            if a >= cap {
                ensure!(
                    g[a] == 0.0,
                    "softmax row {row} col {a} past cap {cap}: {}",
                    g[a]
                );
                continue;
            }
            let d = f64::from(s[a]) - m;
            let want = d.exp() / l;
            let bound = (d.abs() + 24.0) * U32 * want + 1e-38;
            let err = (f64::from(g[a]) - want).abs();
            ensure!(
                err <= bound,
                "softmax row {row} col {a}: {}, f64 {want:e}, bound {bound:.3e}",
                g[a]
            );
            worst = worst.max(err / bound);
            checked += 1;
        }
    }
    println!("dtw_softmax_f32: {checked} values in bound, worst error/bound {worst:.3}");
    Ok(())
}

/// `dtw_norm_f32` against the f64 standardization over each column's first `n` rows.
/// Bound, with `u = 2^-24`, `M` the column's largest `|w|` over those rows and `z` the
/// exact output: the sequential FP32 mean is off by `(n + 1) u M`, so each difference by
/// `(n + 2) u M + u |d|`, the variance relatively by `2 (n + 2) u M / std + (n + 3) u` and
/// the std by half that plus `u`, giving `u ((n + 2) (M / std) (1 + |z|) + (n + 6) |z|)`,
/// doubled for the square root and division roundings
fn norm_case(gpu: &Gpu) -> Result<()> {
    let (heads, batch, t, cols) = (2usize, 2usize, 5usize, 40usize);
    let lens = [3i32, 5];
    let mut rng = Rng::new(75);
    let w: Vec<f32> = (0..heads * batch * t * cols)
        .map(|_| (0.01 * rng.uniform()) as f32)
        .collect();
    let columns = heads * batch * cols;
    let wd = gpu.upload(w.clone())?;
    let ld = gpu.upload(lens.to_vec())?;
    gpu.launch(
        "dtw_norm_f32",
        (blocks(columns, 256), 1, 1),
        (256, 1, 1),
        0,
        &[
            Arg::slice(wd.ptr(), w.len()),
            Arg::slice(ld.ptr(), batch),
            columns.into(),
            t.into(),
            batch.into(),
            cols.into(),
        ],
    )?;
    let got: Vec<f32> = gpu.download(&wd)?;

    let (mut worst, mut checked) = (0f64, 0usize);
    for sb in 0..heads * batch {
        let n = lens[sb % batch] as usize;
        for a in 0..cols {
            let at = |tt: usize| (sb * t + tt) * cols + a;
            let col: Vec<f64> = (0..t).map(|tt| f64::from(w[at(tt)])).collect();
            let mean = col[..n].iter().sum::<f64>() / n as f64;
            let var = col[..n].iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
            let std = var.max(1e-12).sqrt();
            let big = col[..n].iter().fold(0f64, |m, v| m.max(v.abs()));
            for (tt, v) in col.iter().enumerate() {
                let z = (v - mean) / std;
                let nn = n as f64;
                let bound =
                    2.0 * U32 * ((nn + 2.0) * (big / std) * (1.0 + z.abs()) + (nn + 6.0) * z.abs());
                let g = f64::from(got[at(tt)]);
                ensure!(
                    (g - z).abs() <= bound,
                    "norm column ({sb}, {a}) row {tt}: {g}, f64 {z}, bound {bound:.3e}"
                );
                worst = worst.max((g - z).abs() / bound);
                checked += 1;
            }
        }
    }
    println!("dtw_norm_f32: {checked} values in bound, worst error/bound {worst:.3}");
    Ok(())
}

/// The f64 width-7 median with reflect padding at audio position `a`
fn median7(row: &[f32], a: usize) -> f64 {
    let n = row.len() as i64;
    let mut x: Vec<f64> = (-3..=3)
        .map(|o| {
            let i = (a as i64 + o).abs();
            let i = if i >= n { 2 * (n - 1) - i } else { i };
            f64::from(row[i as usize])
        })
        .collect();
    x.sort_by(f64::total_cmp);
    x[3]
}

/// `dtw_median_mean_f32` against exact medians and an f64 mean over heads; the median is
/// an exact selection, so the only roundings are the FP32 sum over `heads` values and the
/// product with the FP32 factor: `(heads + 1) u sum |median| factor`
fn median_case(gpu: &Gpu, width: usize) -> Result<()> {
    let (heads, batch, t, cols) = (3usize, 2usize, 2usize, 50usize);
    let mut rng = Rng::new(76 + width as u64);
    let w: Vec<f32> = (0..heads * batch * t * cols)
        .map(|_| rng.normal() as f32)
        .collect();
    let n_out = batch * t * width;
    let factor = (batch * t * cols) as f32 / (heads * batch * t * cols) as f32;
    let wd = gpu.upload(w.clone())?;
    let out = nan_f32(gpu, n_out)?;
    gpu.launch(
        "dtw_median_mean_f32",
        (blocks(n_out, 256), 1, 1),
        (256, 1, 1),
        0,
        &[
            Arg::slice(out.ptr(), n_out),
            Arg::slice(wd.ptr(), w.len()),
            heads.into(),
            batch.into(),
            t.into(),
            cols.into(),
            width.into(),
            factor.into(),
        ],
    )?;
    let got: Vec<f32> = gpu.download(&out)?;

    let mut worst = 0f64;
    for b in 0..batch {
        for tt in 0..t {
            for a in 0..width {
                let meds: Vec<f64> = (0..heads)
                    .map(|s| median7(&w[((s * batch + b) * t + tt) * cols..][..cols], a))
                    .collect();
                let want = meds.iter().sum::<f64>() * f64::from(factor);
                let mag: f64 = meds.iter().map(|m| m.abs()).sum::<f64>() * f64::from(factor);
                let bound = (heads as f64 + 1.0) * U32 * mag + 1e-38;
                let g = f64::from(got[(b * t + tt) * width + a]);
                ensure!(
                    (g - want).abs() <= bound,
                    "median ({b}, {tt}, {a}) width {width}: {g}, f64 {want}, bound {bound:.3e}"
                );
                worst = worst.max((g - want).abs() / bound);
            }
        }
    }
    println!(
        "dtw_median_mean_f32 width {width}/{cols}: {n_out} values in bound, worst error/bound {worst:.3}"
    );
    Ok(())
}

#[test]
fn decoder_kernels_match_f64() -> Result<()> {
    if !gpu_enabled() {
        return Ok(());
    }
    let gpu = Gpu::new(0)?;
    let cases = [
        // tails in both the 128-row query tile and the 64-key tile
        CausalCase {
            name: "tail",
            batch: 2,
            heads: 3,
            len: 200,
            q_std: 1.0,
        },
        // the longest DTW row with peaked scores, so the running max moves across tiles
        CausalCase {
            name: "peaked 438",
            batch: 1,
            heads: 2,
            len: 438,
            q_std: 3.0,
        },
        // shorter than one key tile
        CausalCase {
            name: "short",
            batch: 2,
            heads: 1,
            len: 5,
            q_std: 1.0,
        },
        // one query row past the first 128-row tile
        CausalCase {
            name: "129",
            batch: 1,
            heads: 1,
            len: 129,
            q_std: 2.0,
        },
    ];
    for (i, c) in cases.iter().enumerate() {
        causal_case(&gpu, c, 40 + i as u64)?;
    }
    embed_case(&gpu, false)?;
    embed_case(&gpu, true)?;
    transpose_case(&gpu)?;
    gather_case(&gpu)?;
    softmax_case(&gpu)?;
    norm_case(&gpu)?;
    median_case(&gpu, 20)?;
    median_case(&gpu, 50)?;
    Ok(())
}

struct Env {
    refs: PathBuf,
    manifest: Value,
    gpu: Gpu,
    blas: Blas,
    attn: DecoderAttn,
    decoder: Decoder,
}

fn env() -> Result<Option<Env>> {
    let (Some(weights), Some(refs)) = (
        std::env::var_os("WHISPERDRZ_WEIGHTS").map(PathBuf::from),
        std::env::var_os("WHISPERDRZ_DECODER_REF").map(PathBuf::from),
    ) else {
        eprintln!(
            "skipping: WHISPERDRZ_WEIGHTS or WHISPERDRZ_DECODER_REF not set (run scripts/gpu-test.sh)"
        );
        return Ok(None);
    };
    let manifest: Value = serde_json::from_slice(&std::fs::read(refs.join("manifest.json"))?)?;
    let gpu = Gpu::new(0)?;
    let blas = Blas::new(&gpu)?;
    let attn = DecoderAttn::new(&gpu)?;
    let mut w = Weights::load(&gpu, &weights, |n| n.starts_with("decoder."))?;
    let decoder = Decoder::new(&gpu, &mut w)?;
    ensure!(w.is_empty(), "{} decoder tensors left unused", w.len());
    Ok(Some(Env {
        refs,
        manifest,
        gpu,
        blas,
        attn,
        decoder,
    }))
}

fn json_list<T>(v: &Value, f: impl Fn(&Value) -> Option<T>) -> Result<Vec<T>> {
    v.as_array()
        .ok_or_else(|| eyre!("{v} is not a list"))?
        .iter()
        .map(|x| f(x).ok_or_else(|| eyre!("bad list item {x}")))
        .collect()
}

/// One phase's inputs from the manifest
struct Phase {
    tokens: Vec<u32>,
    positions: Vec<i32>,
    slots: Vec<usize>,
    q_len: usize,
}

impl Phase {
    fn load(manifest: &Value, name: &str) -> Result<Self> {
        let p = &manifest["phases"][name];
        let tokens = json_list(&p["tokens"], |v| v.as_u64().map(|x| x as u32))?;
        let positions = json_list(&p["positions"], |v| v.as_i64().map(|x| x as i32))?;
        let slots = json_list(&p["slots"], |v| v.as_u64().map(|x| x as usize))?;
        ensure!(
            !slots.is_empty()
                && tokens.len().is_multiple_of(slots.len())
                && positions.len() == tokens.len(),
            "phase {name}: {} tokens, {} positions, {} slots",
            tokens.len(),
            positions.len(),
            slots.len()
        );
        Ok(Self {
            q_len: tokens.len() / slots.len(),
            tokens,
            positions,
            slots,
        })
    }

    fn rows(&self) -> usize {
        self.slots.len()
    }

    fn tokens_n(&self) -> usize {
        self.tokens.len()
    }

    /// The real rows, which lead; padding rows on the `trash` slot trail
    fn real_rows(&self, trash: usize) -> Result<usize> {
        let real = self.slots.iter().take_while(|&&s| s != trash).count();
        ensure!(
            self.slots[real..].iter().all(|&s| s == trash),
            "padding rows must trail the real ones: slots {:?}",
            self.slots
        );
        Ok(real)
    }
}

/// The leading `real` of `rows` equal row blocks of `v`
fn real_part(v: &[f64], rows: usize, real: usize) -> &[f64] {
    &v[..v.len() / rows * real]
}

/// The padding rows' error per layer, which no parity bar judges: the reference discards
/// a padding row too, and its trash slot's contents depend on history
struct PadLog(Vec<(String, Err, Err)>);

impl PadLog {
    /// Logs the trailing padding rows of `ours` and `torch16` against `truth`, which must
    /// be finite in ours
    fn push(
        &mut self,
        at: &str,
        p: &Phase,
        real: usize,
        ours: &[f64],
        torch16: &[f64],
        truth: &[f64],
    ) -> Result<()> {
        if real == p.rows() {
            return Ok(());
        }
        let from = ours.len() / p.rows() * real;
        ensure!(
            ours[from..].iter().all(|v| v.is_finite()),
            "{at}: the padding row is not finite"
        );
        self.0.push((
            at.to_owned(),
            Err::of(&ours[from..], &truth[from..])?,
            Err::of(&torch16[from..], &truth[from..])?,
        ));
        Ok(())
    }

    fn print(&self) {
        let Some(worst) = self
            .0
            .iter()
            .max_by(|a, b| (a.1.max_abs / a.2.max_abs).total_cmp(&(b.1.max_abs / b.2.max_abs)))
        else {
            return;
        };
        println!(
            "padding rows, not judged: finite in all {} layers checked; worst max-abs vs torch FP16 at {}: ours {:.4e} torch {:.4e} ({:.2}x)",
            self.0.len(),
            worst.0,
            worst.1.max_abs,
            worst.2.max_abs,
            worst.1.max_abs / worst.2.max_abs
        );
    }
}

/// Block `i`'s sub-ops, each ours on torch FP16's input to it, as (name, ours, torch FP16,
/// FP32 truth on torch FP16's input) over all rows; `x` is torch FP16's block input
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn sub_ops(
    env: &Env,
    caches: &KvCaches,
    ws: &mut Workspace,
    batch: Batch,
    dir: &std::path::Path,
    i: usize,
    x: &[u16],
) -> Result<Vec<(&'static str, Vec<f64>, Vec<f64>, Vec<f64>)>> {
    let Env {
        gpu,
        blas,
        attn,
        decoder,
        ..
    } = env;
    let nx = x.len();
    let bits = |k: &str| f16_bits(&dir.join(format!("b{i:02}_{k}.npy"))).map(|(_, v)| v);
    let zeros = vec![0u16; nx];
    let mut out = Vec::new();
    // the LayerNorms read x (plus ao, zero here) and write h; the rest read h, write ao
    for (op, name, input, reads_x) in [
        (SubOp::AttnLn, "h1", x.to_vec(), true),
        (SubOp::SelfAttn, "a", bits("h1")?, false),
        (SubOp::CrossLn, "h2", bits("x1")?, true),
        (SubOp::CrossAttn, "c", bits("h2")?, false),
        (SubOp::MlpLn, "h3", bits("x2")?, true),
        (SubOp::Mlp, "m", bits("h3")?, false),
    ] {
        ensure!(
            input.len() == nx,
            "block {i} {name}: input of {} values",
            input.len()
        );
        let (src, dst) = if reads_x {
            (ws.x(), ws.h())
        } else {
            (ws.h(), ws.ao())
        };
        gpu.upload_to(src.ptr(), bytemuck::cast_slice(&input))?;
        gpu.upload_to(dst.ptr(), bytemuck::cast_slice(&vec![NAN16; nx]))?;
        if reads_x {
            gpu.upload_to(ws.ao().ptr(), bytemuck::cast_slice(&zeros))?;
        }
        decoder.sub_op(gpu, blas, attn, caches, i, ws, batch, op)?;
        let dst = if reads_x { ws.h() } else { ws.ao() };
        let ours = bits_f64(&download16(gpu, dst, nx)?);
        let truth = f64s(&dir.join(format!("b{i:02}_{name}32.npy")))?;
        ensure!(
            truth.len() == nx,
            "block {i} {name}: truth of {} values",
            truth.len()
        );
        out.push((name, ours, bits_f64(&bits(name)?), truth));
    }

    Ok(out)
}

/// Where a failing layer's error sits: the element of ours' largest error with ours, torch
/// FP16 and the truth there, then each row's max-abs for ours, torch FP16 and the variants
fn detail(
    title: &str,
    p: &Phase,
    ours: &[f64],
    torch16: &[f64],
    variants: &[(&str, Vec<f64>)],
    truth: &[f64],
) -> Result<()> {
    let cols = ours.len() / p.tokens_n();
    let at = worst_at(ours, truth);
    let (tok, col) = (at / cols, at % cols);
    println!(
        "  {title}: largest error at row {} (slot {}) token {} col {col}: ours {} torch FP16 {} truth {} (FP16 ulp there {:.3e})",
        tok / p.q_len,
        p.slots[tok / p.q_len],
        tok % p.q_len,
        ours[at],
        torch16[at],
        truth[at],
        ulp16(truth[at])
    );
    for r in 0..p.rows() {
        let span = r * p.q_len * cols..(r + 1) * p.q_len * cols;
        let e = |v: &[f64]| Err::of(&v[span.clone()], &truth[span.clone()]).map(|e| e.max_abs);
        let mut line = format!(
            "    row {r} slot {}: max ours {:.4e} torch {:.4e}",
            p.slots[r],
            e(ours)?,
            e(torch16)?
        );
        for (name, v) in variants {
            line += &format!(" {name} {:.4e}", e(v)?);
        }
        println!("{line}");
    }

    Ok(())
}

/// The index of the largest `|a - truth|`
fn worst_at(a: &[f64], truth: &[f64]) -> usize {
    a.iter()
        .zip(truth)
        .map(|(o, t)| (o - t).abs())
        .enumerate()
        .fold(
            (0, -1f64),
            |(bi, bv), (i, d)| if d > bv { (i, d) } else { (bi, bv) },
        )
        .0
}

fn argmax(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0
}

/// Loads torch FP16's self-attention caches and lengths from before `phase` into `caches`:
/// `self_k`/`self_v` `[layers, slots + 1, keep, 1280]` cover the first `keep` positions,
/// every position a phase of these tests reads or writes
fn load_torch_caches(gpu: &Gpu, caches: &KvCaches, dir: &std::path::Path) -> Result<()> {
    let lens = whisperdrz_fixtures::read_npy(&dir.join("lens.npy"))?;
    let ArrayData::I32(lens) = lens.data else {
        bail!("lens.npy is not int32")
    };
    ensure!(
        lens.len() == caches.slots() + 1,
        "{} cache lengths",
        lens.len()
    );
    gpu.upload_to(caches.lens().ptr(), bytemuck::cast_slice(&lens))?;
    for (file, which) in [("self_k.npy", 0), ("self_v.npy", 1)] {
        let (shape, bits) = f16_bits(&dir.join(file))?;
        ensure!(
            shape.len() == 5
                && shape[0] == N_LAYER
                && shape[1] == lens.len()
                && shape[3] * shape[4] == D_MODEL,
            "{file} shape {shape:?}"
        );
        let per = shape[2] * D_MODEL;
        for l in 0..N_LAYER {
            for s in 0..lens.len() {
                let (k, v) = caches.self_slot(l, s)?;
                let src = &bits[(l * lens.len() + s) * per..][..per];
                gpu.upload_to(if which == 0 { k } else { v }, bytemuck::cast_slice(src))?;
            }
        }
    }

    Ok(())
}

/// The DTW forward's timing table, built on the host, against torch's FP16 buffer;
/// returns the failure, if any. It runs after the parity checks so a failure here can't
/// hide theirs
fn timing_table_failures(refs: &std::path::Path) -> Result<Vec<String>> {
    // ours takes correctly rounded exp, sin and cos where torch's CPU uses SLEEF, within an FP32
    // ulp: exp's ulp moves the FP32 argument `t * inv` (t <= 30) by up to 2^-23 relative,
    // which moves sin or cos by up to `30 * 2^-22` absolute with the product's own
    // rounding, and the FP16 rounding can then land on the neighbouring value
    let (_, want) = f16_bits(&refs.join("timing.npy"))?;
    let got: Vec<u16> = decoder::timing_embedding()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    ensure!(
        got.len() == want.len(),
        "timing table of {} values, torch {}",
        got.len(),
        want.len()
    );
    let (mut differ, mut worst, mut worst_ratio) = (0, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(&want) {
        if g != w {
            let (g, w) = (f16_to_f64(g), f16_to_f64(w));
            let bound = 30.0 * 2f64.powi(-22) + ulp16(w);
            differ += 1;
            worst = worst.max((g - w).abs());
            worst_ratio = worst_ratio.max((g - w).abs() / bound);
        }
    }
    println!(
        "timing embedding: {differ} of {} FP16 values differ from torch's, by at most {worst:.3e} (error/bound {worst_ratio:.3})",
        got.len()
    );
    if worst_ratio > 1.0 {
        return Ok(vec![format!(
            "timing embedding {worst:.3e} from torch's, past its bound"
        )]);
    }

    Ok(Vec::new())
}

#[test]
fn decoder_layers_match_torch() -> Result<()> {
    let Some(env) = env()? else { return Ok(()) };
    let Env {
        refs,
        manifest,
        gpu,
        blas,
        attn,
        decoder,
    } = &env;
    let names: Vec<String> = json_list(&manifest["names"], |v| v.as_str().map(str::to_owned))?;
    ensure!(
        names.len() == 35,
        "{} layer names, want embed + 32 blocks + ln + logits",
        names.len()
    );

    let set = TraceSet::open("short3").ok_or_else(|| eyre!("no short3 trace set"))?;
    let (stem, seq) = (
        manifest["stem"].as_str().unwrap_or_default(),
        manifest["seq"].as_u64().unwrap_or_default(),
    );
    let call = set
        .calls(stem, "prefill_flash")?
        .into_iter()
        .find(|e| e.seq == seq)
        .ok_or_else(|| eyre!("{stem}: no prefill_flash #{seq}"))?;
    let xa = set.array(call.arg("xa")?)?;
    let ArrayData::F16(xa) = xa.data else {
        bail!("xa is not float16")
    };
    let xa = gpu.upload(xa.iter().map(|v| v.to_bits()).collect::<Vec<u16>>())?;
    let recorded = set.array(call.ret()?)?;
    let recorded: Vec<f64> = recorded.to_f32()?.into_iter().map(f64::from).collect();

    let slots_n = 5;
    // ours chained from our own prefill; torch FP16's caches for the isolated layers; the
    // fused step path, which must give the chain bit for bit
    let chain_kv = KvCaches::new(gpu, N_LAYER, slots_n)?;
    let iso_kv = KvCaches::new(gpu, N_LAYER, slots_n)?;
    let fused_kv = KvCaches::new(gpu, N_LAYER, slots_n)?;
    let mut ws = Workspace::new(gpu, MAX_ROWS, PREFILL_LEN, 0)?;
    let mut ws_fused = Workspace::new(gpu, MAX_ROWS, PREFILL_LEN, 0)?;

    let env_ref = &env;
    let mut pads = PadLog(Vec::new());
    let mut failed = Vec::new();
    let mut logit_rows = Vec::new();
    for phase in ["prefill", "step1", "step2"] {
        let p = Phase::load(manifest, phase)?;
        // parity is judged on the real rows; padding rows are only logged (see PadLog)
        let real = p.real_rows(slots_n)?;
        let keep = |v: &[f64]| real_part(v, p.rows(), real).to_vec();
        let dir = refs.join(phase);
        let sub_dir = refs.join("sub").join(phase);
        let sub_differs: Vec<u64> = json_list(&manifest["sub_differs"][phase], Value::as_u64)?;
        ensure!(
            sub_differs.len() == N_LAYER && sub_differs.iter().all(|&d| d == 0),
            "{phase}: torch FP16's blocks rebuilt from their sub-ops differ from the recorded ones: {sub_differs:?}"
        );
        let n = p.tokens_n();
        let nx = n * D_MODEL;
        let load = |kind: &str, name: &str| -> Result<Vec<f64>> {
            let v = f64s(&dir.join(format!("{kind}_{name}.npy")))?;
            let want = if name == "logits" { n * N_VOCAB } else { nx };
            ensure!(
                v.len() == want,
                "{phase}/{kind}_{name}: {} values, want {want}",
                v.len()
            );
            Ok(v)
        };
        let load16 = |name: &str| -> Result<Vec<u16>> {
            let (_, b) = f16_bits(&dir.join(format!("f16_{name}.npy")))?;
            Ok(b)
        };
        let upload_x = |buf: &Buf, v: &[u16]| gpu.upload_to(buf.ptr(), bytemuck::cast_slice(v));
        let nan =
            |buf: &Buf, n: usize| gpu.upload_to(buf.ptr(), bytemuck::cast_slice(&vec![NAN16; n]));

        // isolated: our layer N on torch FP16's layer N - 1 output and caches
        if phase == "prefill" {
            decoder.fill_cross(gpu, blas, &iso_kv, &ws, xa.ptr(), &p.slots)?;
            iso_kv.store_xa(gpu, xa.ptr(), &p.slots)?;
        }
        load_torch_caches(gpu, &iso_kv, &dir)?;
        let batch = ws.stage(gpu, &p.tokens, &p.positions, &p.slots)?;
        let (mut iso, mut noise_iso) = (Vec::new(), Vec::new());
        let mut sub_rows: Vec<(usize, Vec<Row>)> = Vec::new();
        let mut prev: Vec<u16> = Vec::new();
        for name in &names {
            let ours: Vec<u16> = match name.as_str() {
                "embed" => {
                    nan(ws.x(), nx)?;
                    decoder.embed(gpu, &ws, batch)?;
                    let ours = download16(gpu, ws.x(), nx)?;
                    // FP16 + FP16 rounded once, exactly what torch computes
                    ensure!(
                        ours == load16("embed")?,
                        "{phase}: embedding differs from torch FP16"
                    );
                    ours
                }
                "ln" => {
                    upload_x(ws.x(), &prev)?;
                    nan(ws.h(), nx)?;
                    decoder.head(gpu, blas, &mut ws, n)?;
                    download16(gpu, ws.h(), nx)?
                }
                "logits" => {
                    upload_x(ws.h(), &prev)?;
                    nan(ws.logits(), n * N_VOCAB)?;
                    decoder.lm_head(blas, &ws, n)?;
                    download16(gpu, ws.logits(), n * N_VOCAB)?
                }
                block => {
                    let i: usize = block.trim_start_matches("block").parse()?;
                    upload_x(ws.x(), &prev)?;
                    decoder.block(gpu, blas, attn, &iso_kv, i, &mut ws, batch)?;
                    download16(gpu, ws.x(), nx)?
                }
            };
            let (ours, torch16, truth) = (bits_f64(&ours), load("f16", name)?, load("iso", name)?);
            pads.push(
                &format!("{phase} isolated {name}"),
                &p,
                real,
                &ours,
                &torch16,
                &truth,
            )?;
            let row = Row::new(name, &keep(&ours), &keep(&torch16), &keep(&truth))?;
            // the yardstick only: the brief allows variant bars for chained max-abs alone
            let v_eff = if name.starts_with("block") {
                let v = load("isoeff", name)?;
                noise_iso.push(Row::new(name, &keep(&v), &keep(&torch16), &keep(&truth))?);
                vec![("eff", v)]
            } else {
                Vec::new()
            };
            // each sub-op of the block on torch FP16's input to it, to place an error
            if let Some(i) = name.strip_prefix("block") {
                let i: usize = i.parse()?;
                let parts = sub_ops(env_ref, &iso_kv, &mut ws, batch, &sub_dir, i, &prev)?;
                let rows = parts
                    .iter()
                    .map(|(n, o, t, tr)| Row::new(n, &keep(o), &keep(t), &keep(tr)))
                    .collect::<Result<Vec<_>>>()?;
                if !row.passes() {
                    report(
                        &format!(
                            "{phase} {name}'s sub-ops on torch FP16's input to each, real rows (truth: FP32 sub-op on that input)"
                        ),
                        &rows,
                        Check::Off,
                    );
                    let at = worst_at(&keep(&ours), &keep(&truth));
                    let cols = D_MODEL;
                    println!(
                        "  at the block's worst element, row {} token {} col {}:",
                        at / cols / p.q_len,
                        at / cols % p.q_len,
                        at % cols
                    );
                    for (n, o, t, tr) in &parts {
                        println!(
                            "    {n:<3} ours {:>14.8} torch FP16 {:>14.8} truth {:>14.8} (FP16 ulp {:.3e})",
                            o[at],
                            t[at],
                            tr[at],
                            ulp16(tr[at])
                        );
                    }
                }
                sub_rows.push((i, rows));
            }
            if !row.passes() {
                detail(
                    &format!("{phase} isolated {name}"),
                    &p,
                    &ours,
                    &torch16,
                    &v_eff,
                    &truth,
                )?;
            }
            iso.push(row);
            prev = load16(name)?;
        }

        // chained: ours from the tokens on our own caches
        if phase == "prefill" {
            decoder.fill_cross(gpu, blas, &chain_kv, &ws, xa.ptr(), &p.slots)?;
            chain_kv.store_xa(gpu, xa.ptr(), &p.slots)?;
            chain_kv.reset_lens(gpu, batch)?;
        }
        let (mut chain, mut noise_qkv, mut noise_sdpa, mut noise_eff) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut chain_logits = Vec::new();
        for name in &names {
            let ours = match name.as_str() {
                "embed" => {
                    nan(ws.x(), nx)?;
                    decoder.embed(gpu, &ws, batch)?;
                    download16(gpu, ws.x(), nx)?
                }
                "ln" => {
                    nan(ws.h(), nx)?;
                    nan(ws.logits(), n * N_VOCAB)?;
                    decoder.head(gpu, blas, &mut ws, n)?;
                    download16(gpu, ws.h(), nx)?
                }
                "logits" => {
                    chain_logits = download16(gpu, ws.logits(), n * N_VOCAB)?;
                    chain_logits.clone()
                }
                block => {
                    let i: usize = block.trim_start_matches("block").parse()?;
                    decoder.block(gpu, blas, attn, &chain_kv, i, &mut ws, batch)?;
                    download16(gpu, ws.x(), nx)?
                }
            };
            let (torch16, truth) = (load("f16", name)?, load("f32", name)?);
            let (v_qkv, v_sdpa) = (load("f16qkv", name)?, load("f16sdpa", name)?);
            let v_eff = load("f16eff", name)?;
            let (t16, tr) = (keep(&torch16), keep(&truth));
            let qkv = Row::new(name, &keep(&v_qkv), &t16, &tr)?;
            let sdpa = Row::new(name, &keep(&v_sdpa), &t16, &tr)?;
            let eff = Row::new(name, &keep(&v_eff), &t16, &tr)?;
            let ours = bits_f64(&ours);
            pads.push(
                &format!("{phase} chained {name}"),
                &p,
                real,
                &ours,
                &torch16,
                &truth,
            )?;
            let row = Row::new(name, &keep(&ours), &t16, &tr)?.bar_from(&[&qkv, &sdpa, &eff]);
            if !row.passes() {
                detail(
                    &format!("{phase} chained {name}"),
                    &p,
                    &ours,
                    &torch16,
                    &[("qkv", v_qkv), ("sdpa", v_sdpa), ("eff", v_eff)],
                    &truth,
                )?;
            }
            chain.push(row);
            noise_qkv.push(qkv);
            noise_sdpa.push(sdpa);
            noise_eff.push(eff);
        }
        chain_kv.advance_lens(gpu, batch)?;

        // the fused prefill / step on its own caches gives the chain bit for bit
        if phase == "prefill" {
            fused_kv.store_xa(gpu, xa.ptr(), &p.slots)?;
            decoder.prefill(
                gpu,
                blas,
                attn,
                &fused_kv,
                &mut ws_fused,
                xa.ptr(),
                &p.tokens,
                &p.slots,
            )?;
        } else {
            let b = ws_fused.stage(gpu, &p.tokens, &p.positions, &p.slots)?;
            decoder.step(gpu, blas, attn, &fused_kv, &mut ws_fused, b)?;
        }
        let fused = download16(gpu, ws_fused.logits(), n * N_VOCAB)?;
        ensure!(
            fused == chain_logits,
            "{phase}: fused step logits differ from the per-layer chain"
        );
        let last = ws_fused.last_logits(gpu, p.rows(), p.q_len)?;
        for r in 0..p.rows() {
            let at = (r * p.q_len + p.q_len - 1) * N_VOCAB;
            ensure!(
                last[r * N_VOCAB..][..N_VOCAB]
                    .iter()
                    .map(|v| v.to_bits())
                    .eq(chain_logits[at..][..N_VOCAB].iter().copied()),
                "{phase}: last_logits row {r} is not the row's last position"
            );
        }
        let ours_lens: Vec<i32> = gpu.download(fused_kv.lens())?;
        let chain_lens: Vec<i32> = gpu.download(chain_kv.lens())?;
        ensure!(
            ours_lens == chain_lens,
            "{phase}: cache lengths {ours_lens:?} vs chain {chain_lens:?}"
        );
        println!("{phase}: fused step == per-layer chain; cache lengths after {ours_lens:?}");

        // logits per row at its last position
        let (torch16, truth) = (load("f16", "logits")?, load("f32", "logits")?);
        let (v_qkv, v_sdpa) = (load("f16qkv", "logits")?, load("f16sdpa", "logits")?);
        let v_eff = load("f16eff", "logits")?;
        let ours = bits_f64(&chain_logits);
        for r in 0..p.rows() {
            let at = (r * p.q_len + p.q_len - 1) * N_VOCAB;
            let seg = |v: &[f64]| v[at..at + N_VOCAB].to_vec();
            let (o, t16, t32) = (seg(&ours), seg(&torch16), seg(&truth));
            let pad = p.slots[r] == slots_n;
            let row = Row::new(
                &format!("{}{}", p.slots[r], if pad { "pad" } else { "" }),
                &o,
                &t16,
                &t32,
            )?;
            let variants = [
                Err::of(&seg(&v_qkv), &t32)?,
                Err::of(&seg(&v_sdpa), &t32)?,
                Err::of(&seg(&v_eff), &t32)?,
            ];
            logit_rows.push((phase, row, variants, argmax(&o), argmax(&t16), argmax(&t32)));
        }
        if phase == "prefill" {
            let direct = Err::of(&ours, &recorded)?;
            println!(
                "prefill logits, ours vs the traced torch run: max {:.4e} mean {:.4e}",
                direct.max_abs, direct.mean_abs
            );
        }

        failed.extend(report(
            &format!("{phase} isolated (truth: FP32 layer on torch FP16's input and caches)"),
            &iso,
            Check::Mean,
        ));
        // the sub-ops carry the full bar: block-level max-abs is reported only, since an FP16
        // ulp flip in one sub-op's output moves the row's LayerNorm statistics and lands in
        // one element of the block output even when every sub-op matches torch
        for (i, rows) in &sub_rows {
            failed.extend(
                rows.iter()
                    .filter(|r| !r.passes())
                    .map(|r| format!("{phase} block{i:02} sub-op {}", r.name)),
            );
        }
        println!(
            "\n{phase} sub-ops on torch FP16's input to each, real rows, asserted at {MAX_RATIO}x: worst block per sub-op"
        );
        for (k, op) in [
            "h1 attn_ln",
            "a self-attention",
            "h2 cross_ln",
            "c cross-attention",
            "h3 mlp_ln",
            "m mlp",
        ]
        .iter()
        .enumerate()
        {
            let ratio =
                |r: &Row, f: fn(&Err) -> f64| f(&r.ours) / f(&r.torch).max(f64::MIN_POSITIVE);
            let worst = |f: fn(&Err) -> f64| {
                sub_rows
                    .iter()
                    .map(|(i, rows)| (*i, ratio(&rows[k], f)))
                    .fold((0, 0f64), |b, c| if c.1 > b.1 { c } else { b })
            };
            let (bm, rm) = worst(|e| e.max_abs);
            let (bn, rn) = worst(|e| e.mean_abs);
            println!(
                "  {op:<18} max-abs {rm:.2}x torch at block{bm:02}, mean-abs {rn:.2}x at block{bn:02}"
            );
        }
        failed.extend(report(
            &format!("{phase} chained (truth: FP32 chain)"),
            &chain,
            Check::Mean,
        ));
        report(
            &format!(
                "{phase} noise yardstick, not asserted: torch FP16 packed QKV in the 'ours' columns"
            ),
            &noise_qkv,
            Check::Off,
        );
        report(
            &format!(
                "{phase} noise yardstick, not asserted: torch FP16 SDPA fallback in the 'ours' columns"
            ),
            &noise_sdpa,
            Check::Off,
        );
        report(
            &format!(
                "{phase} noise yardstick, not asserted: torch FP16 memory-efficient SDPA in the 'ours' columns"
            ),
            &noise_eff,
            Check::Off,
        );
        report(
            &format!(
                "{phase} isolated noise yardstick, not asserted: torch FP16 memory-efficient SDPA layer on torch FP16's input in the 'ours' columns"
            ),
            &noise_iso,
            Check::Off,
        );
    }

    println!(
        "\nlogits per row at its last position (truth: FP32 chain), reported, with argmax asserted; max / mean for ours, torch FP16 and its packed QKV, SDPA and memory-efficient variants, then each one's argmax. '> 1.5x' marks a row past 1.5x torch FP16, which the variants also reach"
    );
    println!(
        "{:<8} {:<6} {:>21} {:>21} {:>21} {:>21} {:>21} {:>7} {:>7} {:>7}",
        "phase",
        "slot",
        "ours",
        "torch",
        "torch qkv",
        "torch sdpa",
        "torch eff",
        "ours",
        "torch",
        "truth"
    );
    let (mut agree16, mut agree32, mut torch_agree32) = (0, 0, 0);
    let pair = |e: &Err| format!("{:.3e}/{:.3e}", e.max_abs, e.mean_abs);
    for (phase, r, v, a_ours, a16, a32) in &logit_rows {
        println!(
            "{:<8} {:<6} {:>21} {:>21} {:>21} {:>21} {:>21} {:>7} {:>7} {:>7}{}",
            phase,
            r.name,
            pair(&r.ours),
            pair(&r.torch),
            pair(&v[0]),
            pair(&v[1]),
            pair(&v[2]),
            a_ours,
            a16,
            a32,
            if r.passes() { "" } else { "  > 1.5x" }
        );
        agree16 += usize::from(a_ours == a16);
        agree32 += usize::from(a_ours == a32);
        torch_agree32 += usize::from(a16 == a32);
        if a_ours != a16 {
            failed.push(format!(
                "logits {phase} row {}: argmax {a_ours} vs torch {a16}",
                r.name
            ));
        }
    }
    println!(
        "argmax: ours = torch FP16 on {agree16}/{n}, ours = FP32 on {agree32}/{n}, torch FP16 = FP32 on {torch_agree32}/{n}",
        n = logit_rows.len()
    );
    pads.print();

    // padding rows can't reach real rows: step1 after the same prefill, with the trash
    // slot's caches and encoder output clean, full of finite garbage at a long length, and
    // full of NaN, and the padding row's token and position changed, gives the real rows
    // bit for bit
    let p0 = Phase::load(manifest, "prefill")?;
    let p1 = Phase::load(manifest, "step1")?;
    let real1 = p1.real_rows(slots_n)?;
    ensure!(real1 < p1.rows(), "step1 has no padding row");
    let mut rng = Rng::new(0x7a54);
    let finite = rng.f16_normal(SELF_CAP.max(CROSS_LEN) * D_MODEL, 0.0, 4.0);
    let nan = vec![NAN16; SELF_CAP.max(CROSS_LEN) * D_MODEL];
    let mut logits_of =
        |kv: &KvCaches, fill: Option<(&[u16], i32)>, tok: u32, pos: i32| -> Result<Vec<u16>> {
            kv.store_xa(gpu, xa.ptr(), &p0.slots)?;
            decoder.prefill(
                gpu,
                blas,
                attn,
                kv,
                &mut ws,
                xa.ptr(),
                &p0.tokens,
                &p0.slots,
            )?;
            let trash = kv.trash();
            if let Some((fill, len)) = fill {
                for l in 0..N_LAYER {
                    for ((k, v), n) in [
                        (kv.self_slot(l, trash)?, SELF_CAP),
                        (kv.cross_slot(l, trash)?, CROSS_LEN),
                    ] {
                        gpu.upload_to(k, bytemuck::cast_slice(&fill[..n * D_MODEL]))?;
                        gpu.upload_to(v, bytemuck::cast_slice(&fill[..n * D_MODEL]))?;
                    }
                }

                // the trash slot's encoder output, which the absorbed cross-attention of
                // large buckets reads for padding rows
                gpu.upload_to(
                    kv.xa_slot(trash)?,
                    bytemuck::cast_slice(&fill[..CROSS_LEN * D_MODEL]),
                )?;
                gpu.upload_to(
                    kv.lens().ptr() + 4 * trash as u64,
                    bytemuck::cast_slice(&[len]),
                )?;
            }
            let (mut toks, mut poss) = (p1.tokens.clone(), p1.positions.clone());
            toks[real1..].fill(tok);
            poss[real1..].fill(pos);
            let b = ws.stage(gpu, &toks, &poss, &p1.slots)?;
            decoder.step(gpu, blas, attn, kv, &mut ws, b)?;
            download16(gpu, ws.logits(), p1.tokens_n() * N_VOCAB)
        };
    // the caches of the per-layer runs above are free now
    let clean = logits_of(&chain_kv, None, 0, 0)?;
    let garbage = logits_of(&iso_kv, Some((&finite, 37)), 50257, 123)?;
    let poisoned = logits_of(&fused_kv, Some((&nan, 5)), 1, 0)?;
    let split = real1 * N_VOCAB;
    let pad_finite = |v: &[u16]| v[split..].iter().all(|&b| f16::from_bits(b).is_finite());
    ensure!(
        pad_finite(&clean),
        "step1 padding row logits not finite on a clean trash slot"
    );
    ensure!(
        garbage[..split] == clean[..split] && poisoned[..split] == clean[..split],
        "step1 real rows change with the trash slot's contents or the padding row's input"
    );
    println!(
        "padding rows: step1's {real1} real rows bitwise equal with the trash slot clean, full of finite garbage at length 37 (padding token 50257, position 123) and full of NaN; padding logits finite: clean {}, garbage {}, NaN {}",
        pad_finite(&clean),
        pad_finite(&garbage),
        pad_finite(&poisoned)
    );
    failed.extend(timing_table_failures(refs)?);
    ensure!(
        failed.is_empty(),
        "worse than {MAX_RATIO}x their torch FP16 bar, or an argmax off torch's: {failed:?}"
    );
    Ok(())
}
