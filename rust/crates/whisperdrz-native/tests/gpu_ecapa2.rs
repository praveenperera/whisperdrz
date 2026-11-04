//! ECAPA2 on the GPU against f64 CPU references and torch
//!
//! - every glue kernel and conv GEMM against an f64 CPU reference on random inputs
//! - per layer against torch's FP32 truth (TF32 off) on three recorded segments, beside
//!   torch's own error in the reference's mode (TF32 convolutions); the spectrogram,
//!   which torch runs on cuFFT in both modes, is measured against an f64 spectrogram
//! - every recorded `_embed_audio_segment` call end to end: cosine with the recorded and
//!   the truth embeddings, with our embeddings written out for the speaker-remap replay
//!   (`rust/tools/ecapa2_remap_check.py`)
//!
//! `rust/tools/ecapa2.py ref` writes the torch side to `WHISPERDRZ_ECAPA2_REF` and
//! `export` the weights `WHISPERDRZ_ECAPA2_WEIGHTS` names; `scripts/gpu-test.sh` runs
//! both through the GPU queue. Skips when either is unset
#![cfg(feature = "cuda")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use eyre::{Result, WrapErr, bail, ensure, eyre};
use safetensors::SafeTensors;
use serde_json::{Value, json};
use whisperdrz_fixtures::{ArrayData, read_npy};
use whisperdrz_native::cuda::device::{Arg, Buf};
use whisperdrz_native::cuda::ecapa2::{
    Ecapa2, Grid, Precision, Se, Steps, conv3x3_layout, conv3x3_layout_f16, dft_basis, normalize,
};
use whisperdrz_native::cuda::{Blas, F32Math, Gpu};

const N_FFT: usize = 511;
const BINS: usize = 256;
const HOP: usize = 160;
const SE_HIDDEN: usize = 128;

/// The reference directory and weights, or `None` to skip
fn inputs() -> Option<(PathBuf, PathBuf)> {
    let dir = std::env::var_os("WHISPERDRZ_ECAPA2_REF").map(PathBuf::from);
    let weights = std::env::var_os("WHISPERDRZ_ECAPA2_WEIGHTS").map(PathBuf::from);
    match (dir, weights) {
        (Some(d), Some(w)) => Some((d, w)),
        _ => {
            eprintln!(
                "skipping: WHISPERDRZ_ECAPA2_REF or WHISPERDRZ_ECAPA2_WEIGHTS not set (run scripts/gpu-test.sh)"
            );
            None
        }
    }
}

fn f32_npy(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let a = read_npy(path)?;
    let ArrayData::F32(v) = a.data else {
        bail!("{} is not float32", path.display())
    };
    Ok((a.shape, v))
}

/// Deterministic uniform values in [-1, 1)
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    fn vec(&mut self, n: usize, scale: f64) -> Vec<f32> {
        (0..n).map(|_| (self.next() * scale) as f32).collect()
    }

    /// A batch norm triple with positive scales
    fn bn(&mut self, c: usize) -> Vec<f32> {
        let mean = self.vec(c, 0.5);
        let k: Vec<f32> = (0..c).map(|_| (1.0 + 0.5 * self.next()) as f32).collect();
        let b = self.vec(c, 0.5);
        [mean, k, b].concat()
    }
}

/// A NaN-filled device buffer of `n` FP32 values, so a kernel that skips an element fails
fn nan(gpu: &Gpu, n: usize) -> Result<Buf> {
    gpu.upload(vec![f32::NAN; n])
}

fn slice(buf: &Buf, n: usize) -> Arg {
    Arg::slice(buf.ptr(), n)
}

/// Kernel-vs-f64 results: name, max-abs error, bound
#[derive(Default)]
struct Checks(Vec<(String, f64, f64)>);

impl Checks {
    /// Records `got` against `want`; NaN or a length mismatch counts as an infinite error
    fn check(&mut self, name: &str, got: &[f32], want: &[f64], bound: f64) {
        let err = if got.len() == want.len() {
            got.iter()
                .zip(want)
                .map(|(&g, &w)| (g as f64 - w).abs())
                .map(|d| if d.is_nan() { f64::INFINITY } else { d })
                .fold(0.0, f64::max)
        } else {
            f64::INFINITY
        };
        self.0.push((name.to_string(), err, bound));
    }

    fn finish(self) -> Result<()> {
        println!("{:<34} {:>12} {:>12}", "kernel vs f64", "max-abs", "bound");
        let mut failed = Vec::new();
        for (name, err, bound) in &self.0 {
            let ok = err <= bound;
            println!(
                "{name:<34} {err:>12.3e} {bound:>12.3e} {}",
                if ok { "ok" } else { "FAIL" }
            );
            if !ok {
                failed.push(name.clone());
            }
        }
        ensure!(
            failed.is_empty(),
            "kernels off their f64 references: {failed:?}"
        );
        Ok(())
    }
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

/// Reflect index without edge repeat, as torch pads
fn reflect(i: isize, len: usize) -> usize {
    let len = len as isize;
    let i = i.abs();
    (if i >= len { 2 * (len - 1) - i } else { i }) as usize
}

/// Trunk buffer `[h + 2][c][w + 2]` from torch layout `[c][h][w]`, zero border
fn to_grid(x: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let g = Grid::trunk(h, c, w);
    let mut out = vec![0f32; g.len()];
    for ci in 0..c {
        for r in 0..h {
            for t in 0..w {
                out[((r + 1) * c + ci) * g.cols + t + 1] = x[(ci * h + r) * w + t];
            }
        }
    }

    out
}

/// f64 trunk buffer from torch layout, zero border
fn to_grid64(x: &[f64], c: usize, h: usize, w: usize) -> Vec<f64> {
    let g = Grid::trunk(h, c, w);
    let mut out = vec![0f64; g.len()];
    for ci in 0..c {
        for r in 0..h {
            for t in 0..w {
                out[((r + 1) * c + ci) * g.cols + t + 1] = x[(ci * h + r) * w + t];
            }
        }
    }

    out
}

/// f64 3x3 conv, padding 1, stride `s` along rows, no bias, torch layouts; also the
/// sum of absolute terms per output, the scale of its rounding error
fn conv3x3_ref(
    x: &[f32],
    w: &[f32],
    (ci, co, h, wd, s): (usize, usize, usize, usize, usize),
) -> (Vec<f64>, Vec<f64>) {
    let ho = h / s;
    let (mut y, mut mag) = (vec![0f64; co * ho * wd], vec![0f64; co * ho * wd]);
    for o in 0..co {
        for r in 0..ho {
            for t in 0..wd {
                let (mut acc, mut abs) = (0f64, 0f64);
                for c in 0..ci {
                    for kh in 0..3 {
                        for kw in 0..3 {
                            let (hr, tc) = ((r * s + kh) as isize - 1, (t + kw) as isize - 1);
                            if hr < 0 || hr >= h as isize || tc < 0 || tc >= wd as isize {
                                continue;
                            }
                            let p = w[((o * ci + c) * 3 + kh) * 3 + kw] as f64
                                * x[(c * h + hr as usize) * wd + tc as usize] as f64;
                            acc += p;
                            abs += p.abs();
                        }
                    }
                }
                y[(o * ho + r) * wd + t] = acc;
                mag[(o * ho + r) * wd + t] = abs;
            }
        }
    }

    (y, mag)
}

fn bn64(v: f64, bn: &[f32], c: usize, channels: usize) -> f64 {
    (v - bn[c] as f64) * bn[channels + c] as f64 + bn[2 * channels + c] as f64
}

/// Bound for a sum: FP32's unit roundoff times the summed magnitude, with headroom for
/// the reduction order and the final roundings
fn sum_bound(mag: f64) -> f64 {
    8.0 * f32::EPSILON as f64 * mag
}

#[test]
fn kernels_match_f64() -> Result<()> {
    if inputs().is_none() {
        return Ok(());
    }
    let gpu = Gpu::new(0)?;
    let blas = Blas::new(&gpu)?;
    let steps = Steps {
        gpu: &gpu,
        blas: &blas,
    };
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut checks = Checks::default();

    // frames: reflect-padded, windowed, the 400-tap window centered in 511
    let samples = 3001;
    let audio = rng.vec(samples, 1.0);
    let window = rng.vec(400, 1.0);
    let frames = 1 + (samples - 1) / HOP;
    let (a_buf, w_buf) = (gpu.upload(audio.clone())?, gpu.upload(window.clone())?);
    let f_buf = nan(&gpu, frames * N_FFT)?;
    ensure!(
        steps.frames(&f_buf, &a_buf, samples, &w_buf)? == frames,
        "frame count"
    );
    let mut want = vec![0f64; frames * N_FFT];
    for f in 0..frames {
        for n in 55..455 {
            let x = audio[reflect((f * HOP + n) as isize - 255, samples)];
            want[f * N_FFT + n] = x as f64 * window[n - 55] as f64;
        }
    }
    let got: Vec<f32> = gpu.download(&f_buf)?;
    checks.check("ecapa_frames", &got, &want, f32::EPSILON as f64);

    // DFT as one FP32 GEMM, from the frames just made
    let basis = gpu.upload(dft_basis())?;
    let spec = nan(&gpu, frames * 2 * BINS)?;
    steps.dft(&spec, &f_buf, &basis, frames)?;
    let (mut want, mut mag) = (vec![0f64; frames * 2 * BINS], 0f64);
    for f in 0..frames {
        let row = &got[f * N_FFT..(f + 1) * N_FFT];
        mag = mag.max(row.iter().map(|v| v.abs() as f64).sum());
        for k in 0..BINS {
            let (mut re, mut im) = (0f64, 0f64);
            for (n, &x) in row.iter().enumerate() {
                let a = std::f64::consts::TAU * ((k * n) % N_FFT) as f64 / N_FFT as f64;
                re += x as f64 * a.cos();
                im -= x as f64 * a.sin();
            }
            want[f * 2 * BINS + k] = re;
            want[f * 2 * BINS + BINS + k] = im;
        }
    }
    let got: Vec<f32> = gpu.download(&spec)?;
    checks.check("dft gemm (fp32)", &got, &want, sum_bound(mag));

    // log power
    let y = rng.vec(frames * 2 * BINS, 3.0);
    let y_buf = gpu.upload(y.clone())?;
    let lp = nan(&gpu, BINS * frames)?;
    steps.log_power(&lp, &y_buf, frames)?;
    let mut want = vec![0f64; BINS * frames];
    for k in 0..BINS {
        for f in 0..frames {
            let (re, im) = (
                y[f * 2 * BINS + k] as f64,
                y[f * 2 * BINS + BINS + k] as f64,
            );
            want[k * frames + f] = (re * re + im * im + 1e-6).ln();
        }
    }
    let got: Vec<f32> = gpu.download(&lp)?;
    // logf near a power of 1e-6 amplifies the power's rounding; 4 ulps of the log's scale
    checks.check(
        "ecapa_log_power",
        &got,
        &want,
        4.0 * f32::EPSILON as f64 * max_abs(&want),
    );

    // center rows into the padded trunk input
    let (h, w) = (37, 300);
    let x = rng.vec(h * w, 4.0);
    let x_buf = gpu.upload(x.clone())?;
    let g = Grid::trunk(h, 1, w);
    let c_buf = nan(&gpu, g.len())?;
    ensure!(
        steps.center_rows(&c_buf, &x_buf, (h, w), 4)? == g,
        "center_rows grid"
    );
    let mut want = vec![0f64; h * w];
    for r in 0..h {
        let row = &x[r * w..(r + 1) * w];
        let mean = row.iter().map(|&v| v as f64).sum::<f64>() / w as f64;
        for t in 0..w {
            want[r * w + t] = row[t] as f64 - mean;
        }
    }
    let got: Vec<f32> = gpu.download(&c_buf)?;
    checks.check(
        "ecapa_center_rows",
        &got[..g.len()],
        &to_grid64(&want, 1, h, w),
        sum_bound(4.0) + 4.0 * f32::EPSILON as f64 * 8.0,
    );

    // 3x3 convs at both strides with their fused epilogue (bias, ReLU, batch norm,
    // border), with scratch for every output row and for two rows at a time (chunks); a
    // width that is not a multiple of 4 exercises the padded columns
    for (stride, chunk) in [(1, None), (2, None), (1, Some(2)), (2, Some(3))] {
        let (ci, co, h, w) = (6, 10, 12, 13);
        let x = rng.vec(ci * h * w, 1.0);
        let wt = rng.vec(co * ci * 9, 1.0);
        let (bias, bn) = (rng.vec(co, 0.5), rng.bn(co));
        let g = Grid::trunk(h, ci, w);
        let x_buf = gpu.upload(to_grid(&x, ci, h, w))?;
        let w_buf = gpu.upload(conv3x3_layout(&wt, co, ci))?;
        let (b_buf, bn_buf) = (gpu.upload(bias.clone())?, gpu.upload(bn.clone())?);
        let og = Grid::trunk(h / stride, co, w);
        let y_buf = nan(&gpu, og.len())?;
        let rows = chunk.unwrap_or(h / stride);
        let scratch = nan(&gpu, rows * 3 * co * og.cols)?;
        let got_g = steps.conv3x3(
            F32Math::Full,
            (&x_buf, g),
            &scratch,
            (&y_buf, false),
            &w_buf,
            (&b_buf, &bn_buf),
            co,
            stride,
        )?;
        ensure!(got_g == og, "conv3x3 grid {got_g:?}");
        let (pre, mag) = conv3x3_ref(&x, &wt, (ci, co, h, w, stride));
        let ho = h / stride;
        let mut post = vec![0f64; pre.len()];
        let mut bound = 0f64;
        for o in 0..co {
            for j in 0..ho * w {
                let v = (pre[o * ho * w + j] + bias[o] as f64).max(0.0);
                post[o * ho * w + j] = bn64(v, &bn, o, co);
                // the sum's rounding through the batch norm scale, then the epilogue's own
                bound = bound.max(
                    sum_bound(mag[o * ho * w + j] + bias[o].abs() as f64) * bn[co + o] as f64
                        + 4.0 * f32::EPSILON as f64 * post[o * ho * w + j].abs(),
                );
            }
        }
        let got: Vec<f32> = gpu.download(&y_buf)?;
        checks.check(
            &format!("conv3x3 + epilogue (fp32, stride {stride}, chunk {chunk:?})"),
            &got[..og.len()],
            &to_grid64(&post, co, ho, w),
            bound,
        );
    }

    // the FP16-input conv with an FP16 output and with an FP32 output, on 8-aligned
    // columns: operands rounded to FP16, then exact products and FP32 sums, so it is
    // checked against the f64 conv of the rounded operands
    for (stride, out_f16) in [(1, true), (2, false)] {
        let (ci, co, h, w) = (6, 10, 12, 13);
        let to_h = |v: &[f32]| -> Vec<f32> {
            v.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect()
        };
        let x = to_h(&rng.vec(ci * h * w, 1.0));
        let wt = to_h(&rng.vec(co * ci * 9, 1.0));
        let (bias, bn) = (rng.vec(co, 0.5), rng.bn(co));
        let g = Grid::trunk_aligned(h, ci, w, 8);
        ensure!(g.cols.is_multiple_of(8), "fp16 grid {g:?}");
        let host_x = to_grid(&x, ci, h, w);
        // to_grid lays out 4-aligned rows; re-pack into the 8-aligned grid
        let g4 = Grid::trunk(h, ci, w);
        let mut xh = vec![0u16; g.len()];
        for r in 0..g.rows {
            for c in 0..ci {
                for t in 0..g4.cols.min(g.cols) {
                    xh[(r * ci + c) * g.cols + t] =
                        half::f16::from_f32(host_x[(r * ci + c) * g4.cols + t]).to_bits();
                }
            }
        }
        // ci 6 pads each 18-value weight row to 24, as 164 channels pad 492 to 496
        let wh = conv3x3_layout_f16(&conv3x3_layout(&wt, co, ci), co, ci);
        let (x_buf, w_buf) = (gpu.upload(xh)?, gpu.upload(wh)?);
        let (b_buf, bn_buf) = (gpu.upload(bias.clone())?, gpu.upload(bn.clone())?);
        let ho = h / stride;
        let og = Grid::trunk_aligned(ho, co, w, 8);
        let y_buf = nan(&gpu, og.len())?;
        let scratch = nan(&gpu, ho * 3 * co * og.cols)?;
        let got_g = steps.conv3x3(
            F32Math::F16Inputs,
            (&x_buf, g),
            &scratch,
            (&y_buf, out_f16),
            &w_buf,
            (&b_buf, &bn_buf),
            co,
            stride,
        )?;
        ensure!(got_g == og, "fp16 conv3x3 grid {got_g:?}");
        let (pre, mag) = conv3x3_ref(&x, &wt, (ci, co, h, w, stride));
        let (mut post, mut bound) = (vec![0f64; pre.len()], 0f64);
        for o in 0..co {
            for j in 0..ho * w {
                let v = (pre[o * ho * w + j] + bias[o] as f64).max(0.0);
                post[o * ho * w + j] = bn64(v, &bn, o, co);
                let out_round = if out_f16 {
                    2f64.powi(-11)
                } else {
                    4.0 * f32::EPSILON as f64
                };
                bound = bound.max(
                    sum_bound(mag[o * ho * w + j] + bias[o].abs() as f64) * bn[co + o] as f64
                        + out_round * post[o * ho * w + j].abs(),
                );
            }
        }
        let got: Vec<f32> = if out_f16 {
            let bits: Vec<u16> = gpu.download(&y_buf)?;
            bits[..og.len()]
                .iter()
                .map(|&b| half::f16::from_bits(b).to_f32())
                .collect()
        } else {
            let v: Vec<f32> = gpu.download(&y_buf)?;
            v[..og.len()].to_vec()
        };
        // the whole buffer against the expected grid, so the border and the padding
        // columns must come out zero
        let mut want = vec![0f64; og.len()];
        for o in 0..co {
            for r in 0..ho {
                for t in 0..w {
                    want[((r + 1) * co + o) * og.cols + t + 1] = post[(o * ho + r) * w + t];
                }
            }
        }
        checks.check(
            &format!("conv3x3 f16 inputs + epilogue (stride {stride}, fp16 out {out_f16})"),
            &got,
            &want,
            bound,
        );
    }

    // the plain epilogue (downsample and head convs) on a trunk buffer
    {
        let (co, h, w) = (10, 8, 13);
        let og = Grid::trunk(h, co, w);
        let pre = rng.vec(co * h * w, 1.0);
        let (bias, bn) = (rng.vec(co, 0.5), rng.bn(co));
        let mut host = to_grid(&pre, co, h, w);
        // junk on the border, which the epilogue must zero
        for (i, v) in host.iter_mut().enumerate() {
            if *v == 0.0 {
                *v = i as f32;
            }
        }
        let y_buf = gpu.upload(host)?;
        let (b_buf, bn_buf) = (gpu.upload(bias.clone())?, gpu.upload(bn.clone())?);
        steps.epilogue(
            &y_buf,
            0,
            og,
            slice(&b_buf, co),
            slice(&bn_buf, 3 * co),
            true,
        )?;
        let mut post = vec![0f64; pre.len()];
        for o in 0..co {
            for j in 0..h * w {
                let v = (pre[o * h * w + j] as f64 + bias[o] as f64).max(0.0);
                post[o * h * w + j] = bn64(v, &bn, o, co);
            }
        }
        let got: Vec<f32> = gpu.download(&y_buf)?;
        checks.check(
            "ecapa_conv_epilogue",
            &got[..og.len()],
            &to_grid64(&post, co, h, w),
            4.0 * f32::EPSILON as f64 * max_abs(&post),
        );
    }

    // 1x1 stride-2 downsample conv
    {
        let (ci, co, h, w) = (7, 9, 10, 11);
        let x = rng.vec(ci * h * w, 1.0);
        let wt = rng.vec(co * ci, 1.0);
        let g = Grid::trunk(h, ci, w);
        let x_buf = gpu.upload(to_grid(&x, ci, h, w))?;
        let w_buf = gpu.upload(wt.clone())?;
        let og = Grid::trunk(h / 2, co, w);
        let y_buf = nan(&gpu, og.len())?;
        steps.conv1x1_s2(F32Math::Full, &x_buf, g, &y_buf, &w_buf, co)?;
        let (mut want, mut mag) = (vec![0f64; co * (h / 2) * w], 0f64);
        for o in 0..co {
            for r in 0..h / 2 {
                for t in 0..w {
                    let terms =
                        (0..ci).map(|c| wt[o * ci + c] as f64 * x[(c * h + 2 * r) * w + t] as f64);
                    want[(o * (h / 2) + r) * w + t] = terms.clone().sum();
                    mag = mag.max(terms.map(f64::abs).sum());
                }
            }
        }
        let got: Vec<f32> = gpu.download(&y_buf)?;
        checks.check(
            "conv1x1 stride-2 gemm (fp32)",
            &og.to_torch(&got),
            &want,
            sum_bound(mag),
        );
    }

    // head 1x1 conv1d reading a strided view
    {
        let (ci, co, frames, ld) = (37, 11, 23, 29);
        let x = rng.vec(ci * ld, 1.0);
        let wt = rng.vec(co * ci, 1.0);
        let (x_buf, w_buf) = (gpu.upload(x.clone())?, gpu.upload(wt.clone())?);
        let y_buf = nan(&gpu, co * frames)?;
        steps.conv1d(
            F32Math::Full,
            (x_buf.at(3 * 4), ld),
            y_buf.ptr(),
            w_buf.ptr(),
            (ci, co, frames),
            0.0,
        )?;
        let (mut want, mut mag) = (vec![0f64; co * frames], 0f64);
        for o in 0..co {
            for t in 0..frames {
                let terms = (0..ci).map(|c| wt[o * ci + c] as f64 * x[c * ld + 3 + t] as f64);
                want[o * frames + t] = terms.clone().sum();
                mag = mag.max(terms.map(f64::abs).sum());
            }
        }
        let got: Vec<f32> = gpu.download(&y_buf)?;
        checks.check(
            "conv1d gemm (fp32, strided input)",
            &got,
            &want,
            sum_bound(mag),
        );
    }

    // channel mean over a trunk buffer and a head buffer
    for (h, c, w, pad) in [(9, 13, 300, 1), (1, 21, 333, 0)] {
        let g = if pad == 1 {
            Grid::trunk(h, c, w)
        } else {
            Grid::head(c, w)
        };
        let x = rng.vec(c * h * w, 2.0);
        let host = if pad == 1 {
            to_grid(&x, c, h, w)
        } else {
            x.clone()
        };
        let x_buf = gpu.upload(host)?;
        let m_buf = nan(&gpu, c)?;
        steps.channel_mean(&m_buf, &x_buf, g)?;
        let want: Vec<f64> = (0..c)
            .map(|ci| {
                x[ci * h * w..(ci + 1) * h * w]
                    .iter()
                    .map(|&v| v as f64)
                    .sum::<f64>()
                    / (h * w) as f64
            })
            .collect();
        let got: Vec<f32> = gpu.download(&m_buf)?;
        checks.check(
            &format!("ecapa_channel_mean (pad {pad})"),
            &got,
            &want,
            sum_bound(2.0),
        );
    }

    // squeeze-excitation MLP
    {
        let c = 40;
        let mean = rng.vec(c, 1.0);
        let (w1, b1, bn) = (
            rng.vec(SE_HIDDEN * c, 0.3),
            rng.vec(SE_HIDDEN, 0.3),
            rng.bn(SE_HIDDEN),
        );
        let (w2, b2) = (rng.vec(c * SE_HIDDEN, 0.3), rng.vec(c, 0.3));
        let se = Se::upload(
            &gpu,
            w1.clone(),
            b1.clone(),
            bn.clone(),
            w2.clone(),
            b2.clone(),
        )?;
        let m_buf = gpu.upload(mean.clone())?;
        let s_buf = nan(&gpu, c)?;
        steps.se_fc(&s_buf, &m_buf, &se)?;
        let hidden: Vec<f64> = (0..SE_HIDDEN)
            .map(|r| {
                let z = b1[r] as f64
                    + (0..c)
                        .map(|i| w1[r * c + i] as f64 * mean[i] as f64)
                        .sum::<f64>();
                bn64(z.max(0.0), &bn, r, SE_HIDDEN)
            })
            .collect();
        let want: Vec<f64> = (0..c)
            .map(|o| {
                let s = b2[o] as f64
                    + (0..SE_HIDDEN)
                        .map(|r| w2[o * SE_HIDDEN + r] as f64 * hidden[r])
                        .sum::<f64>();
                1.0 / (1.0 + (-s).exp())
            })
            .collect();
        let got: Vec<f32> = gpu.download(&s_buf)?;
        // sigmoid's slope is at most 1/4, so the logit's sum bound carries over
        checks.check(
            "ecapa_se_fc",
            &got,
            &want,
            sum_bound(0.3 * (c as f64 + SE_HIDDEN as f64 * 8.0)),
        );
    }

    // rescale plus residual, with and without the residual
    for with_r in [true, false] {
        let (h, c, w) = (5, 6, 17);
        let g = Grid::trunk(h, c, w);
        let x = rng.vec(c * h * w, 2.0);
        let r = rng.vec(c * h * w, 2.0);
        let s = rng.vec(c, 1.0);
        let (x_buf, r_buf, s_buf) = (
            gpu.upload(to_grid(&x, c, h, w))?,
            gpu.upload(to_grid(&r, c, h, w))?,
            gpu.upload(s.clone())?,
        );
        let o_buf = nan(&gpu, g.len())?;
        let h_buf = gpu.upload(vec![0x7e00u16; g.len()])?;
        steps.scale_residual(
            (&o_buf, with_r.then_some(&h_buf)),
            &x_buf,
            &s_buf,
            with_r.then_some(&r_buf),
            g,
        )?;
        let want: Vec<f64> = (0..c * h * w)
            .map(|i| x[i] as f64 * s[i / (h * w)] as f64 + if with_r { r[i] as f64 } else { 0.0 })
            .collect();
        let got: Vec<f32> = gpu.download(&o_buf)?;
        checks.check(
            &format!("ecapa_scale_residual (residual {with_r})"),
            &got,
            &to_grid64(&want, c, h, w),
            2.0 * f32::EPSILON as f64 * 4.0,
        );
        if with_r {
            // the FP16 copy: the FP32 output rounded to nearest, half an FP16 ulp
            let halves: Vec<u16> = gpu.download(&h_buf)?;
            let halves: Vec<f32> = halves
                .iter()
                .map(|&b| half::f16::from_bits(b).to_f32())
                .collect();
            let want = to_grid64(&want, c, h, w);
            let bound = want.iter().fold(0f64, |m, v| m.max(v.abs())) * 2f64.powi(-11) + 1e-7;
            checks.check(
                "ecapa_scale_residual_f16 (fp16 copy)",
                &halves,
                &want,
                bound,
            );
        }
    }

    // Res2Net group input with reflect padding
    for with_b in [true, false] {
        let (c, frames, pad) = (5, 9, 2);
        let (a, b) = (rng.vec(c * frames, 1.0), rng.vec(c * frames, 1.0));
        let (a_buf, b_buf) = (gpu.upload(a.clone())?, gpu.upload(b.clone())?);
        let width = frames + 2 * pad;
        let o_buf = nan(&gpu, c * width)?;
        let b_arg = if with_b {
            slice(&b_buf, c * frames)
        } else {
            Arg::EMPTY
        };
        steps.reflect_add(&o_buf, slice(&a_buf, c * frames), b_arg, (c, frames, pad))?;
        let mut want = vec![0f64; c * width];
        for ci in 0..c {
            for j in 0..width {
                let src = ci * frames + reflect(j as isize - pad as isize, frames);
                want[ci * width + j] = a[src] as f64 + if with_b { b[src] as f64 } else { 0.0 };
            }
        }
        let got: Vec<f32> = gpu.download(&o_buf)?;
        checks.check(
            &format!("ecapa_reflect_add (second input {with_b})"),
            &got,
            &want,
            f32::EPSILON as f64 * 2.0,
        );
    }

    // global context for attentive pooling, then the pooling itself
    {
        let (c, frames) = (7, 300);
        let x: Vec<f32> = rng.vec(c * frames, 1.0).iter().map(|v| v + 0.5).collect();
        let x_buf = gpu.upload(x.clone())?;
        let o_buf = nan(&gpu, 3 * c * frames)?;
        steps.stats_cat(&o_buf, &x_buf, c, frames)?;
        let mut want = vec![0f64; 3 * c * frames];
        for ci in 0..c {
            let row: Vec<f64> = x[ci * frames..(ci + 1) * frames]
                .iter()
                .map(|&v| v as f64)
                .collect();
            let mean = row.iter().sum::<f64>() / frames as f64;
            let var =
                row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (frames - 1) as f64;
            for t in 0..frames {
                want[ci * frames + t] = row[t];
                want[(c + ci) * frames + t] = mean;
                want[(2 * c + ci) * frames + t] = (var + 1e-5).sqrt();
            }
        }
        let got: Vec<f32> = gpu.download(&o_buf)?;
        checks.check("ecapa_stats_cat", &got, &want, sum_bound(1.5));

        let logits = rng.vec(c * frames, 4.0);
        let bn = rng.bn(2 * c);
        let (l_buf, bn_buf) = (gpu.upload(logits.clone())?, gpu.upload(bn.clone())?);
        let p_buf = nan(&gpu, 2 * c)?;
        steps.attentive_pool(&p_buf, &l_buf, &x_buf, &bn_buf, c, frames)?;
        let mut want = vec![0f64; 2 * c];
        for ci in 0..c {
            let l = &logits[ci * frames..(ci + 1) * frames];
            let max = l.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v as f64));
            let e: Vec<f64> = l.iter().map(|&v| (v as f64 - max).exp()).collect();
            let sum: f64 = e.iter().sum();
            let (mut mu, mut sq) = (0f64, 0f64);
            for t in 0..frames {
                let xs = e[t] / sum * x[ci * frames + t] as f64;
                mu += xs;
                sq += xs * x[ci * frames + t] as f64;
            }
            let sigma = ((sq - mu * mu).abs() + 1e-5).sqrt();
            want[2 * ci] = bn64(mu, &bn, ci, 2 * c);
            want[2 * ci + 1] = bn64(sigma, &bn, c + ci, 2 * c);
        }
        let got: Vec<f32> = gpu.download(&p_buf)?;
        // sigma's sqrt of a difference near mu^2 amplifies rounding; the batch norm scales
        // by up to 1.5
        checks.check(
            "ecapa_attentive_pool",
            &got,
            &want,
            1.5 * sum_bound(1.5 * 1.5) * 8.0,
        );
    }

    checks.finish()
}

/// The ECAPA2 spectrogram (`log(|STFT|^2 + 1e-6)` minus its mean over time) in f64
fn spec_f64(audio: &[f32], window: &[f32]) -> Vec<f64> {
    let frames = 1 + (audio.len() - 1) / HOP;
    let (cos, sin): (Vec<f64>, Vec<f64>) = (0..N_FFT)
        .map(|i| {
            let a = std::f64::consts::TAU * i as f64 / N_FFT as f64;
            (a.cos(), a.sin())
        })
        .unzip();
    let mut out = vec![0f64; BINS * frames];
    let mut frame = vec![0f64; N_FFT];
    for f in 0..frames {
        for (n, v) in frame.iter_mut().enumerate() {
            *v = if (55..455).contains(&n) {
                audio[reflect((f * HOP + n) as isize - 255, audio.len())] as f64
                    * window[n - 55] as f64
            } else {
                0.0
            };
        }
        for k in 0..BINS {
            let (mut re, mut im) = (0f64, 0f64);
            for (n, &x) in frame.iter().enumerate() {
                let i = (k * n) % N_FFT;
                re += x * cos[i];
                im -= x * sin[i];
            }
            out[k * frames + f] = (re * re + im * im + 1e-6).ln();
        }
    }
    for row in out.chunks_mut(frames) {
        let mean = row.iter().sum::<f64>() / frames as f64;
        row.iter_mut().for_each(|v| *v -= mean);
    }

    out
}

fn window(weights: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(weights)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let view = st.tensor("pre_processing.0.window")?;
    Ok(view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}

fn errors(got: &[f32], truth: &[f64]) -> (f64, f64) {
    let d: Vec<f64> = got
        .iter()
        .zip(truth)
        .map(|(&g, &t)| (g as f64 - t).abs())
        .map(|d| if d.is_nan() { f64::INFINITY } else { d })
        .collect();
    (
        d.iter().fold(0.0, |m, &v| m.max(v)),
        d.iter().sum::<f64>() / d.len() as f64,
    )
}

fn precision_name(p: Precision) -> &'static str {
    match p {
        Precision::Fp32 => "fp32",
        Precision::Reference => "reference (tf32 convs)",
        Precision::F16Convs => "f16 conv inputs",
    }
}

#[test]
fn layers_match_torch_fp32() -> Result<()> {
    let Some((dir, weights)) = inputs() else {
        return Ok(());
    };
    let manifest: Value = serde_json::from_slice(&std::fs::read(dir.join("manifest.json"))?)?;
    let taps: Vec<&str> = manifest["taps"]
        .as_array()
        .ok_or_else(|| eyre!("no taps"))?
        .iter()
        .filter_map(Value::as_str)
        .collect();
    let layers = manifest["layers"]
        .as_object()
        .ok_or_else(|| eyre!("no layers"))?;
    ensure!(
        layers.len() == 3 && taps.len() == 26,
        "manifest has {} layer segments, {} taps",
        layers.len(),
        taps.len()
    );
    let window = window(&weights)?;

    let gpu = Gpu::new(0)?;
    let blas = Blas::new(&gpu)?;
    let mut failed = Vec::new();
    for precision in [Precision::Fp32, Precision::Reference, Precision::F16Convs] {
        let model = Ecapa2::load(&gpu, &weights, precision)?;
        println!(
            "\nper-layer error against torch FP32 (TF32 off), ours in {}",
            precision_name(precision)
        );
        println!(
            "{:<32} {:<10} {:>11} {:>11} {:>11} {:>11} {:>7} {:>7}",
            "segment", "tap", "ours max", "torch max", "ours mean", "torch mean", "max x", "mean x"
        );
        for (seg, rows) in layers {
            let (_, audio) = f32_npy(&dir.join("segments").join(format!("{seg}.npy")))?;
            let mut got = Vec::new();
            model.forward(&gpu, &blas, &audio, Some(&mut got))?;
            let names: Vec<&str> = got.iter().map(|t| t.name).collect();
            ensure!(names == taps, "{seg}: taps {names:?}");

            for tap in &got {
                let row = &rows[tap.name];
                let (shape, truth) = f32_npy(
                    &dir.join("layers")
                        .join(seg)
                        .join(format!("{}.npy", tap.name)),
                )?;
                ensure!(
                    shape == tap.shape,
                    "{seg} {}: shape {:?}, torch {shape:?}",
                    tap.name,
                    tap.shape
                );
                let truth64: Vec<f64> = truth.iter().map(|&v| v as f64).collect();
                let (ours, torch) = if tap.name == "spec" {
                    // torch runs the spectrogram on cuFFT in both modes, so its
                    // reference-mode error is zero by construction; measure ours and
                    // torch's cuFFT against an f64 spectrogram instead
                    let exact = spec_f64(&audio, &window);
                    (errors(&tap.data, &exact), errors(&truth, &exact))
                } else {
                    let r = &row["reference_mode"];
                    let get = |k: &str| {
                        r[k].as_f64()
                            .ok_or_else(|| eyre!("{seg} {}: no {k}", tap.name))
                    };
                    (
                        errors(&tap.data, &truth64),
                        (get("max_abs")?, get("mean_abs")?),
                    )
                };
                let ratio = (ours.0 / torch.0, ours.1 / torch.1);
                let ok = ours.0 <= 1.5 * torch.0 && ours.1 <= 1.5 * torch.1;
                println!(
                    "{seg:<32} {:<10} {:>11.3e} {:>11.3e} {:>11.3e} {:>11.3e} {:>7.3} {:>7.3} {}",
                    tap.name,
                    ours.0,
                    torch.0,
                    ours.1,
                    torch.1,
                    ratio.0,
                    ratio.1,
                    if ok { "ok" } else { "OVER" }
                );
                // the gate applies to the mode the runtime ships (FP32); the TF32 mode
                // is reported beside it
                if !ok && precision == Precision::Fp32 {
                    failed.push(format!("{seg} {}", tap.name));
                }
            }
        }
    }
    ensure!(
        failed.is_empty(),
        "layers over 1.5x torch's error: {failed:?}"
    );
    Ok(())
}

/// Cosine similarity, failing on a length mismatch, a zero or non-finite norm, or a
/// non-finite result, so a zero or NaN embedding can't slip past a min over scores
fn cosine(a: &[f32], b: &[f64]) -> Result<f64> {
    ensure!(
        a.len() == b.len(),
        "cosine of {} and {} values",
        a.len(),
        b.len()
    );
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y).sum();
    let na = a.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
    let nb = b.iter().map(|y| y * y).sum::<f64>().sqrt();
    ensure!(
        na.is_finite() && na > 0.0 && nb.is_finite() && nb > 0.0,
        "embedding norms {na} and {nb}"
    );
    let c = dot / (na * nb);
    ensure!(c.is_finite(), "cosine {c}");
    Ok(c)
}

fn floats(v: &Value) -> Result<Vec<f64>> {
    v.as_array()
        .ok_or_else(|| eyre!("not an array"))?
        .iter()
        .map(|x| x.as_f64().ok_or_else(|| eyre!("not a number")))
        .collect()
}

#[test]
fn embeddings_match_recorded() -> Result<()> {
    let Some((dir, weights)) = inputs() else {
        return Ok(());
    };
    let manifest: Value = serde_json::from_slice(&std::fs::read(dir.join("manifest.json"))?)?;
    let segments = manifest["segments"]
        .as_array()
        .ok_or_else(|| eyre!("no segments"))?;
    ensure!(
        segments.len() == 76,
        "manifest has {} segments, want 76",
        segments.len()
    );

    let gpu = Gpu::new(0)?;
    let blas = Blas::new(&gpu)?;
    let mut out = BTreeMap::new();
    let mut fp32_min_truth = 0.0;
    for precision in [Precision::Fp32, Precision::Reference, Precision::F16Convs] {
        let model = Ecapa2::load(&gpu, &weights, precision)?;
        let mut ours = BTreeMap::new();
        let (mut rec, mut tru) = (Vec::new(), Vec::new());
        for seg in segments {
            let name = seg["name"]
                .as_str()
                .ok_or_else(|| eyre!("segment without a name"))?;
            let audio_path = dir.join(
                seg["audio"]
                    .as_str()
                    .ok_or_else(|| eyre!("{name}: no audio"))?,
            );
            let (_, audio) = f32_npy(&audio_path)?;
            ensure!(
                audio.len() as u64 == seg["samples"].as_u64().unwrap_or(0),
                "{name}: sample count"
            );
            let e = normalize(&model.embed(&gpu, &blas, &audio)?);
            ensure!(e.len() == 192, "{name}: embedding width {}", e.len());
            rec.push(
                cosine(&e, &floats(&seg["recorded"])?)
                    .wrap_err_with(|| format!("{name} vs recorded"))?,
            );
            tru.push(
                cosine(&e, &floats(&seg["truth"])?).wrap_err_with(|| format!("{name} vs truth"))?,
            );
            ours.insert(
                name.to_string(),
                json!({"set": seg["set"], "stem": seg["stem"], "seq": seg["seq"], "embedding": e}),
            );
        }
        // an empty set would leave the min at infinity and pass the floor below
        ensure!(!rec.is_empty(), "no recorded segments embedded");
        let stat = |v: &[f64]| {
            (
                v.iter().fold(f64::INFINITY, |m, &x| m.min(x)),
                v.iter().sum::<f64>() / v.len() as f64,
            )
        };
        let (r, t) = (stat(&rec), stat(&tru));
        println!(
            "{:<24} cos vs recorded min {:.9} mean {:.9}; vs truth min {:.9} mean {:.9}",
            precision_name(precision),
            r.0,
            r.1,
            t.0,
            t.1
        );
        if precision == Precision::Fp32 {
            fp32_min_truth = t.0;
        }
        out.insert(
            match precision {
                Precision::Fp32 => "fp32",
                Precision::Reference => "reference",
                Precision::F16Convs => "f16convs",
            },
            ours,
        );
    }

    let path = std::env::var_os("WHISPERDRZ_ECAPA2_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| dir.join("ours.json"));
    std::fs::write(&path, serde_json::to_vec(&out)?)?;
    println!("wrote {}", path.display());
    // a sanity floor, not the gate: FP32 against the FP32 truth sits within float noise,
    // so anything this far off is a broken layer
    ensure!(
        fp32_min_truth > 0.9999,
        "FP32 embeddings off the FP32 truth: min cos {fp32_min_truth}"
    );
    Ok(())
}
