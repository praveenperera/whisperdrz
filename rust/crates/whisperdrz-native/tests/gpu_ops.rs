//! The shared transformer kernels (`cuda::ops`) against f64 references on their exact
//! FP16 inputs
//!
//! Each output is checked element by element against a bound derived from the kernel's
//! own approximations (one FP16 rounding of the result, plus the FP16 rounding of P that
//! FlashAttention-2 makes), never against a tolerance tuned to pass. Outputs are NaN-filled
//! before each launch and every comparison fails on non-finite values. Skips unless
//! `WHISPERDRZ_GPU=1`; run through the GPU queue with `rust/scripts/gpu-test.sh`
#![cfg(feature = "cuda")]

#[allow(dead_code)]
#[path = "support/numerics.rs"]
mod numerics;

use eyre::{Result, ensure};
use numerics::{Err, NAN16, Rng, f16_round, f16_to_f64, gpu_enabled, ulp16};
use whisperdrz_native::cuda::Gpu;
use whisperdrz_native::cuda::ops::{self, HeadView, LN_EPS};

fn to_f64(bits: &[u16]) -> Vec<f64> {
    bits.iter().copied().map(f16_to_f64).collect()
}

/// `erfc(z)` for `z >= 0` in f64: the Maclaurin series below 3 (terms stay under e^9, so
/// about 12 digits survive) and the Laplace continued fraction above
fn erfc_pos(z: f64) -> f64 {
    if z < 3.0 {
        let (mut term, mut sum, mut n) = (z, z, 0.0);
        while term.abs() > 1e-17 * sum.abs() {
            n += 1.0;
            term *= -z * z / n;
            sum += term / (2.0 * n + 1.0);
        }
        return 1.0 - sum * 2.0 / std::f64::consts::PI.sqrt();
    }

    let mut f = z;
    for k in (1..200).rev() {
        f = z + (k as f64 / 2.0) / f;
    }
    (-z * z).exp() / std::f64::consts::PI.sqrt() / f
}

fn gelu64(x: f64) -> f64 {
    let y = x / std::f64::consts::SQRT_2;
    // 1 + erf(y), without cancellation for negative y
    let one_plus_erf = if y >= 0.0 {
        2.0 - erfc_pos(y)
    } else {
        erfc_pos(-y)
    };
    0.5 * x * one_plus_erf
}

fn layernorm64(x: &[f64], w: &[f64], b: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let rstd = 1.0 / (var + LN_EPS as f64).sqrt();
    x.iter()
        .zip(w.iter().zip(b))
        .map(|(v, (w, b))| (v - mean) * rstd * w + b)
        .collect()
}

/// Rows of LayerNorm input: mostly N(0.3, 2), plus rows with a large mean (variance
/// from a small spread on a big offset) and a near-constant row
fn ln_input(rng: &mut Rng, rows: usize, cols: usize) -> Vec<u16> {
    let mut x = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let (mean, std) = match r % 5 {
            1 => (100.0, 1.0),
            3 => (-3.0, 0.01),
            _ => (0.3, 2.0),
        };
        x.extend(rng.f16_normal(cols, mean, std));
    }
    x
}

/// Checks LayerNorm output against f64 element by element, within one FP16 spacing of
/// the reference (half from the final rounding) plus the error FP32 statistics can carry
/// into `x - mean`: a sum of `k` FP32 terms is off by at most `k * 2^-24 * sum |x|`, and
/// each lane adds `8 * ceil(cols / 256)` values before a 5-level shuffle tree, so the
/// mean's error is at most `gamma * mean |x|`, scaled by `|w| * rstd` in the output. On
/// rows with a large mean that term passes one ulp of outputs near zero, as it does for
/// torch's FP32 Welford statistics. The variance's error is relative (`gamma / 2` on
/// rstd) and stays far inside the ulp term
fn ln_check(what: &str, got: &[u16], x: &[f64], w: &[f64], b: &[f64], cols: usize) -> Result<Err> {
    let gamma = ((cols / 8).div_ceil(32) * 8 + 5) as f64 * 2f64.powi(-24);
    let got = to_f64(got);
    let mut want = Vec::with_capacity(x.len());
    let (mut bad, mut ulps) = (Vec::new(), 0f64);
    for (r, row) in x.chunks(cols).enumerate() {
        let y = layernorm64(row, w, b);
        let n = cols as f64;
        let mean = row.iter().sum::<f64>() / n;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
        let rstd = 1.0 / (var + LN_EPS as f64).sqrt();
        let mean_err = gamma * row.iter().map(|v| v.abs()).sum::<f64>() / n;
        for (c, &yc) in y.iter().enumerate() {
            let g = got[r * cols + c];
            ulps = ulps.max((g - yc).abs() / ulp16(yc));
            if (g - yc).abs() > ulp16(yc) + w[c].abs() * rstd * mean_err && bad.len() < 5 {
                bad.push((r, c, g, yc));
            }
        }
        want.extend(y);
    }
    ensure!(
        bad.is_empty(),
        "{what}: outside the bound at (row, col, got, want) {bad:?}"
    );
    println!("{what}: worst error {ulps:.2} FP16 ulps");
    Err::of(&got, &want)
}

#[test]
fn layernorm_matches_f64() -> Result<()> {
    if !gpu_enabled() {
        return Ok(());
    }
    let gpu = Gpu::new(0)?;
    let mut rng = Rng::new(1);
    let mut checks = 0;
    // 1280 is the model width; 392 leaves the last chunk of most lanes empty
    for (rows, cols) in [(37usize, 1280usize), (10, 392), (3, 2048)] {
        let x = ln_input(&mut rng, rows, cols);
        let w = rng.f16_normal(cols, 1.0, 0.2);
        let b = rng.f16_normal(cols, 0.0, 0.5);
        let add = rng.f16_normal(rows * cols, 0.0, 1.0);
        let (xd, wd, bd, addd) = (
            gpu.upload(x.clone())?,
            gpu.upload(w.clone())?,
            gpu.upload(b.clone())?,
            gpu.upload(add.clone())?,
        );
        let out = gpu.upload(vec![NAN16; rows * cols])?;
        ops::layernorm(&gpu, xd.ptr(), wd.ptr(), bd.ptr(), out.ptr(), (rows, cols))?;
        let got: Vec<u16> = gpu.download(&out)?;

        let (w64, b64) = (to_f64(&w), to_f64(&b));
        let err = ln_check(
            &format!("layernorm {rows}x{cols}"),
            &got,
            &to_f64(&x),
            &w64,
            &b64,
            cols,
        )?;
        println!(
            "layernorm {rows}x{cols}: max {:.3e} mean {:.3e}",
            err.max_abs, err.mean_abs
        );
        checks += 1;

        // fused residual add: x becomes f16(x + add) bit for bit, then the LayerNorm of it
        let out = gpu.upload(vec![NAN16; rows * cols])?;
        ops::add_layernorm(
            &gpu,
            xd.ptr(),
            addd.ptr(),
            wd.ptr(),
            bd.ptr(),
            out.ptr(),
            (rows, cols),
        )?;
        let (sum, got): (Vec<u16>, Vec<u16>) = (gpu.download(&xd)?, gpu.download(&out)?);
        let want_sum: Vec<f64> = to_f64(&x)
            .iter()
            .zip(to_f64(&add))
            .map(|(a, d)| f16_round(a + d))
            .collect();
        ensure!(
            to_f64(&sum) == want_sum,
            "add_layernorm {rows}x{cols}: residual sum differs"
        );
        let err = ln_check(
            &format!("add_layernorm {rows}x{cols}"),
            &got,
            &want_sum,
            &w64,
            &b64,
            cols,
        )?;
        println!(
            "add_layernorm {rows}x{cols}: max {:.3e} mean {:.3e}",
            err.max_abs, err.mean_abs
        );
        checks += 1;
    }

    ensure!(checks == 6, "{checks} LayerNorm checks ran");
    Ok(())
}

#[test]
fn gelu_and_add_match_f64() -> Result<()> {
    if !gpu_enabled() {
        return Ok(());
    }
    let gpu = Gpu::new(0)?;
    let mut rng = Rng::new(2);
    let n = 8 * 4099;
    let mut x = rng.f16_normal(n, 0.0, 2.5);
    // the whole FP16 range of interest, both tails and zero
    for (i, v) in [-12.0, -8.0, -5.0, -1e-3, 0.0, 1e-3, 5.0, 60000.0]
        .into_iter()
        .enumerate()
    {
        x[i] = half::f16::from_f64(v).to_bits();
    }

    let xd = gpu.upload(x.clone())?;
    ops::gelu_inplace(&gpu, xd.ptr(), n)?;
    let got = to_f64(&gpu.download::<u16>(&xd)?);
    let x64 = to_f64(&x);
    let want: Vec<f64> = x64.iter().map(|&v| gelu64(v)).collect();
    // torch's formula, x * 0.5 * (1 + erf), cancels in FP32 for large negative x, which
    // this kernel shares on purpose: allow FP32 erf's error near 1 on top of one ulp
    let bad: Vec<_> = (0..n)
        .filter(|&i| (got[i] - want[i]).abs() > ulp16(want[i]) + x64[i].abs() * 2f64.powi(-22))
        .take(5)
        .map(|i| (x64[i], got[i], want[i]))
        .collect();
    ensure!(
        bad.is_empty(),
        "gelu outside its bound at (x, got, want) {bad:?}"
    );
    let err = Err::of(&got, &want)?;
    println!(
        "gelu {n}: max {:.3e} mean {:.3e}",
        err.max_abs, err.mean_abs
    );

    let d = rng.f16_normal(n, 0.0, 3.0);
    let (xd, dd) = (gpu.upload(x.clone())?, gpu.upload(d.clone())?);
    ops::add_inplace(&gpu, xd.ptr(), dd.ptr(), n)?;
    let got = to_f64(&gpu.download::<u16>(&xd)?);
    let want: Vec<f64> = x64
        .iter()
        .zip(to_f64(&d))
        .map(|(a, b)| f16_round(a + b))
        .collect();
    ensure!(got == want, "residual add differs from f16(x + d)");
    Ok(())
}

/// One attention case's geometry
struct AttnCase {
    name: &'static str,
    batch: usize,
    heads: usize,
    lq: usize,
    lk: usize,
    /// q, k and v packed in one `[batch, l, 3 * heads * 64]` buffer, as the encoder's QKV
    /// GEMM writes them (needs lq == lk); else a padded q buffer and a packed kv buffer
    packed: bool,
    q_std: f64,
}

/// Runs one case and checks every output element against the exact f64 attention with
/// the bound `0.5 ulp16(o) + (2^-11 + 1e-4) * sum_j p_j |v_j|`: FP16 rounding of the
/// output, FP16 rounding of each softmax weight in P @ V (relative error at most 2^-11),
/// and 1e-4 relative for FP32 accumulation, `ex2.approx` and the FP32 scores
fn attention_case(gpu: &Gpu, c: &AttnCase, seed: u64) -> Result<()> {
    let mut rng = Rng::new(seed);
    let (hd, scale) = (c.heads * 64, 0.125f64);
    // buffers and views: element (b, i, h, d) of each operand
    let (q_ld, k_ld, v_ld, q_off, k_off, v_off, q_buf, kv_buf);
    if c.packed {
        assert_eq!(c.lq, c.lk);
        q_ld = 3 * hd;
        (k_ld, v_ld) = (q_ld, q_ld);
        (q_off, k_off, v_off) = (0, hd, 2 * hd);
        q_buf = rng.f16_normal(c.batch * c.lq * q_ld, 0.0, 1.0);
        kv_buf = Vec::new();
    } else {
        q_ld = hd + 16;
        (k_ld, v_ld) = (2 * hd, 2 * hd);
        (q_off, k_off, v_off) = (0, 0, hd);
        q_buf = rng.f16_normal(c.batch * c.lq * q_ld, 0.0, 1.0);
        kv_buf = rng.f16_normal(c.batch * c.lk * k_ld, 0.0, 1.0);
    }
    // scale q up for peaked softmaxes
    let q_buf: Vec<u16> = q_buf
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let col = i % q_ld;
            if col >= q_off && col < q_off + hd {
                half::f16::from_f64(f16_to_f64(v) * c.q_std).to_bits()
            } else {
                v
            }
        })
        .collect();
    let kv_src = if c.packed { &q_buf } else { &kv_buf };
    let o_ld = hd + 8;

    let qd = gpu.upload(q_buf.clone())?;
    let kvd = if c.packed {
        None
    } else {
        Some(gpu.upload(kv_buf.clone())?)
    };
    let kv_ptr = kvd.as_ref().map_or(qd.ptr(), |b| b.ptr());
    let od = gpu.upload(vec![NAN16; c.batch * c.lq * o_ld])?;
    let view = |ptr: u64, rows: usize, ld: usize, off: usize| HeadView {
        ptr: ptr + 2 * off as u64,
        len: c.batch * rows * ld - off,
        ld,
        batch_stride: rows * ld,
    };
    ops::attention(
        gpu,
        view(od.ptr(), c.lq, o_ld, 0),
        view(qd.ptr(), c.lq, q_ld, q_off),
        view(kv_ptr, c.lk, k_ld, k_off),
        view(kv_ptr, c.lk, v_ld, v_off),
        (c.batch, c.heads, c.lq, c.lk),
        scale as f32,
    )?;
    let o: Vec<u16> = gpu.download(&od)?;

    let q64 = to_f64(&q_buf);
    let kv64 = to_f64(kv_src);
    let (mut got, mut want, mut worst) = (Vec::new(), Vec::new(), 0f64);
    let mut s = vec![0f64; c.lk];
    for b in 0..c.batch {
        for h in 0..c.heads {
            for i in 0..c.lq {
                let qrow = &q64[(b * c.lq + i) * q_ld + q_off + h * 64..][..64];
                for (j, sj) in s.iter_mut().enumerate() {
                    let krow = &kv64[(b * c.lk + j) * k_ld + k_off + h * 64..][..64];
                    *sj = scale * qrow.iter().zip(krow).map(|(a, b)| a * b).sum::<f64>();
                }
                let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let l: f64 = s.iter().map(|v| (v - m).exp()).sum();
                for d in 0..64 {
                    let (mut acc, mut mag) = (0f64, 0f64);
                    for (j, sj) in s.iter().enumerate() {
                        let p = (sj - m).exp() / l;
                        let vj = kv64[(b * c.lk + j) * v_ld + v_off + h * 64 + d];
                        acc += p * vj;
                        mag += p * vj.abs();
                    }
                    let g = f16_to_f64(o[(b * c.lq + i) * o_ld + h * 64 + d]);
                    let bound = 0.5 * ulp16(acc) + (2f64.powi(-11) + 1e-4) * mag;
                    ensure!(
                        (g - acc).abs() <= bound,
                        "{}: o[{b}, {i}, {h}, {d}] = {g}, f64 {acc}, bound {bound:.3e}",
                        c.name
                    );
                    worst = worst.max((g - acc).abs() / bound);
                    got.push(g);
                    want.push(acc);
                }
            }
        }
    }

    // the 8 padding columns of every output row stay as they were: nothing wrote past
    // the heads
    for row in o.chunks(o_ld) {
        ensure!(
            row[hd..].iter().all(|&v| v == NAN16),
            "{}: wrote into o's row padding",
            c.name
        );
    }
    let err = Err::of(&got, &want)?;
    println!(
        "attention {}: B={} H={} lq={} lk={}: max {:.3e} mean {:.3e}, worst error/bound {worst:.3}",
        c.name, c.batch, c.heads, c.lq, c.lk, err.max_abs, err.mean_abs
    );
    Ok(())
}

#[test]
fn attention_matches_f64() -> Result<()> {
    if !gpu_enabled() {
        return Ok(());
    }
    let gpu = Gpu::new(0)?;
    let cases = [
        // tails in both the 128-row query tile and the 64-key tile, packed like the encoder
        AttnCase {
            name: "packed tail",
            batch: 2,
            heads: 3,
            lq: 200,
            lk: 200,
            packed: true,
            q_std: 1.0,
        },
        // the encoder's length with peaked rows, so the online max moves across 24 tiles
        AttnCase {
            name: "peaked 1500",
            batch: 1,
            heads: 2,
            lq: 1500,
            lk: 1500,
            packed: true,
            q_std: 3.0,
        },
        // the decoder's cross-attention shape: few queries over the audio frames
        AttnCase {
            name: "cross",
            batch: 2,
            heads: 2,
            lq: 37,
            lk: 1500,
            packed: false,
            q_std: 1.0,
        },
        // fewer keys than one tile, query rows spilling into a second block
        AttnCase {
            name: "short keys",
            batch: 1,
            heads: 1,
            lq: 130,
            lk: 5,
            packed: false,
            q_std: 1.0,
        },
    ];
    for (i, c) in cases.iter().enumerate() {
        attention_case(&gpu, c, 10 + i as u64)?;
    }
    Ok(())
}
