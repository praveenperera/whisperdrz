//! Times the cuBLASLt algorithms for the decode step's linears, the measurement behind
//! `blas/decode.rs`'s table
//!
//! For each decode linear shape (packed QKV, the 1280x1280 projections, fc1, fc2) and
//! each graph bucket's row count it times torch's heuristic choice, every heuristic
//! result under the decode workspace and their split-K variants. Weights rotate through
//! a 256 MiB pool, past the 96 MiB L2, since the decode step streams 1.6 GB of weights
//! and reads each one cold; activations stay hot as they do in the step. A device copy
//! of the weight's bytes, read half and written half, gives a practical floor. Each
//! timed batch queues behind a long copy, so the GPU's time is measured rather than the
//! host's launch rate. Every candidate's output is checked for bit equality between two
//! runs and against the exact f64 product: a candidate qualifies only with its max-abs
//! error within 1.25x and its mean-abs error within 1.05x torch's choice's, since some
//! kernels round to FP16 before the bias add and land up to 2 ulps from torch, which the
//! per-sub-op bars of `gpu_decoder` reject. Skips unless `WHISPERDRZ_GEMM_PROBE` names the
//! JSON report to write; GPU only, so run it through the queue
#![cfg(feature = "cuda")]

#[allow(dead_code)]
#[path = "support/numerics.rs"]
mod numerics;

use std::path::PathBuf;

use eyre::{Result, ensure};
use half::f16;
use numerics::Rng;
use serde_json::json;
use whisperdrz_native::cuda::blas::{LtAlgo, LtCandidate};
use whisperdrz_native::cuda::{Blas, Gpu};

const SHAPES: [(&str, usize, usize); 4] = [
    ("qkv", 1280, 3840),
    ("proj", 1280, 1280),
    ("fc1", 1280, 5120),
    ("fc2", 5120, 1280),
];
const ROWS: [usize; 9] = [1, 2, 4, 8, 16, 32, 42, 52, 54];
const POOL: usize = 256 << 20;
const LAUNCHES: usize = 64;
const TRIALS: usize = 7;

struct Bench<'a> {
    gpu: &'a Gpu,
    /// Two scratch buffers for the copy that holds the GPU busy while a batch is queued
    busy: (u64, u64),
}

impl Bench<'_> {
    /// Median over trials of the GPU time per launch of `launch(i)`, `i` the launch index
    fn time_us(&self, mut launch: impl FnMut(usize) -> Result<()>) -> Result<f64> {
        // warm: plan creation, module loads, clocks
        for i in 0..8 {
            launch(i)?;
        }
        self.gpu.synchronize()?;

        let mut trials = Vec::with_capacity(TRIALS);
        for t in 0..TRIALS {
            self.gpu.copy_dtod(self.busy.0, self.busy.1, POOL)?;
            let ms = self.gpu.time_ms(|| {
                for i in 0..LAUNCHES {
                    launch(t * LAUNCHES + i)?;
                }
                Ok(())
            })?;
            trials.push(ms as f64 * 1e3 / LAUNCHES as f64);
        }
        trials.sort_by(f64::total_cmp);
        Ok(trials[TRIALS / 2])
    }
}

fn max_diff(a: &[u16], b: &[u16]) -> (f64, usize) {
    let mut worst = 0f64;
    let mut differ = 0;
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f16::from_bits(*x).to_f64(), f16::from_bits(*y).to_f64());
        let d = (x - y).abs();
        if d.is_nan() {
            return (f64::INFINITY, a.len());
        }
        if d > 0.0 {
            differ += 1;
        }
        worst = worst.max(d);
    }
    (worst, differ)
}

/// The exact `x @ w^T + b` in f64 for the first `rows` rows, `[rows, n_out]`
fn truth(x: &[u16], w: &[u16], b: &[u16], (rows, n_in, n_out): (usize, usize, usize)) -> Vec<f64> {
    let h = |v: u16| f16::from_bits(v).to_f64();
    let (xf, wf): (Vec<f64>, Vec<f64>) = (
        x[..rows * n_in].iter().map(|&v| h(v)).collect(),
        w[..n_out * n_in].iter().map(|&v| h(v)).collect(),
    );
    let mut out = vec![0f64; rows * n_out];
    let threads = 32;
    let per = n_out.div_ceil(threads);
    let cols: Vec<Vec<(usize, Vec<f64>)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let (xf, wf) = (&xf, &wf);
                scope.spawn(move || {
                    (t * per..((t + 1) * per).min(n_out))
                        .map(|n| {
                            let wr = &wf[n * n_in..(n + 1) * n_in];
                            let col = (0..rows)
                                .map(|r| {
                                    let xr = &xf[r * n_in..(r + 1) * n_in];
                                    xr.iter().zip(wr).map(|(a, b)| a * b).sum::<f64>() + h(b[n])
                                })
                                .collect();
                            (n, col)
                        })
                        .collect()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (n, col) in cols.into_iter().flatten() {
        for (r, v) in col.into_iter().enumerate() {
            out[r * n_out + n] = v;
        }
    }
    out
}

/// Max-abs and mean-abs error of FP16 outputs against the exact values
fn error(got: &[u16], truth: &[f64]) -> (f64, f64) {
    let mut max = 0f64;
    let mut sum = 0f64;
    for (g, t) in got.iter().zip(truth) {
        let d = (f16::from_bits(*g).to_f64() - t).abs();
        max = if d.is_nan() {
            f64::INFINITY
        } else {
            max.max(d)
        };
        sum += d;
    }
    (max, sum / got.len() as f64)
}

fn algo_json(a: &LtAlgo) -> serde_json::Value {
    json!({
        "id": a.id, "tile": a.tile, "stages": a.stages, "splitk": a.splitk,
        "reduction": a.reduction, "swizzle": a.swizzle, "custom": a.custom,
        "inner": a.inner, "cluster": a.cluster,
    })
}

#[test]
fn probe_decode_gemms() -> Result<()> {
    let Some(out) = std::env::var_os("WHISPERDRZ_GEMM_PROBE").map(PathBuf::from) else {
        eprintln!("skipping: WHISPERDRZ_GEMM_PROBE not set");
        return Ok(());
    };
    let gpu = Gpu::new(0)?;
    let blas = Blas::new(&gpu)?;
    let mut rng = Rng::new(7);

    // one random 16 MiB block tiled over the pool: values don't change the timing
    let block = 8 << 20;
    let seed = gpu.upload(rng.f16_normal(block, 0.0, 0.02))?;
    let pool = gpu.alloc(POOL)?;
    for off in (0..POOL).step_by(block * 2) {
        gpu.copy_dtod(seed.ptr(), pool.at(off), block * 2)?;
    }
    let max_in = 5120;
    let x = gpu.upload(rng.f16_normal(54 * max_in, 0.0, 1.0))?;
    let bias = gpu.upload(rng.f16_normal(max_in, 0.0, 0.1))?;
    let y = gpu.alloc(54 * max_in * 2)?;
    let busy = (gpu.alloc(POOL)?, gpu.alloc(POOL)?);
    let bench = Bench {
        gpu: &gpu,
        busy: (busy.0.ptr(), busy.1.ptr()),
    };
    let copy_dst = gpu.alloc(64 << 20)?;
    // weight copy 0 is the start of the seed block, every shape's weights fit in it
    let (w_host, x_host, b_host): (Vec<u16>, Vec<u16>, Vec<u16>) = (
        gpu.download(&seed)?,
        gpu.download(&x)?,
        gpu.download(&bias)?,
    );

    let mut report = Vec::new();
    let mut table = Vec::new();
    println!(
        "{:5} {:>4} {:>9} {:>9} {:>9} {:>9} {:>7} {:>7}  best",
        "shape", "rows", "torch us", "best us", "copy us", "1.67 us", "torchGB", "bestGB"
    );
    for (name, n_in, n_out) in SHAPES {
        let wbytes = n_in * n_out * 2;
        let copies = POOL / wbytes;
        let w_at = |i: usize| pool.at((i % copies) * wbytes);
        // practical floor: move the weight's bytes once, half read and half written
        let copy_us = bench.time_us(|i| gpu.copy_dtod(w_at(i), copy_dst.ptr(), wbytes / 2))?;
        let ideal_us = wbytes as f64 / 1.67e6;
        let exact = truth(&x_host, &w_host, &b_host, (54, n_in, n_out));

        for rows in ROWS {
            let dims = (rows, n_in, n_out);
            let n = rows * n_out;
            let run_torch =
                |i: usize| blas.linear_bias(x.ptr(), w_at(i), bias.ptr(), y.ptr(), dims);
            run_torch(0)?;
            let want: Vec<u16> = gpu.download(&y)?;
            let want = want[..n].to_vec();
            let exact = &exact[..n];
            let torch_err = error(&want, exact);
            let torch_us = bench.time_us(run_torch)?;

            let cands: Vec<LtCandidate> =
                blas.linear_bias_candidates((x.ptr(), w_at(0), y.ptr()), dims, 64)?;
            let mut rows_json = Vec::new();
            let mut best: Option<(f64, LtAlgo, (f64, f64))> = None;
            for c in &cands {
                let run = |i: usize| {
                    blas.linear_bias_with(x.ptr(), w_at(i), bias.ptr(), y.ptr(), dims, c.algo)
                };
                gpu.memset_zero(y.ptr(), n * 2)?;
                if let Err(err) = run(0) {
                    rows_json
                        .push(json!({"algo": algo_json(&c.algo), "error": format!("{err:#}")}));
                    continue;
                }
                let first: Vec<u16> = gpu.download(&y)?;
                gpu.memset_zero(y.ptr(), n * 2)?;
                run(0)?;
                let second: Vec<u16> = gpu.download(&y)?;
                let deterministic = first[..n] == second[..n];
                let (diff, differ) = max_diff(&first[..n], &want);
                let err = error(&first[..n], exact);
                let us = bench.time_us(run)?;
                // FP32 sums in another order move a correctly rounded output by at most
                // an ulp from torch's and leave its error statistics where torch's are
                let ok =
                    deterministic && err.0 <= 1.25 * torch_err.0 && err.1 <= 1.05 * torch_err.1;
                if ok && best.is_none_or(|(b, _, _)| us < b) {
                    best = Some((us, c.algo, err));
                }
                rows_json.push(json!({
                    "algo": algo_json(&c.algo), "us": us, "workspace": c.workspace,
                    "heuristic": c.heuristic, "waves": c.waves, "deterministic": deterministic,
                    "max_diff": diff, "differ": differ, "max_err": err.0, "mean_err": err.1, "ok": ok,
                }));
            }

            let (best_us, best_algo, best_err) = best.unwrap_or((
                f64::NAN,
                LtAlgo {
                    id: -1,
                    ..LtAlgo::default()
                },
                (f64::NAN, f64::NAN),
            ));
            println!(
                "{name:5} {rows:4} {torch_us:9.2} {best_us:9.2} {copy_us:9.2} {ideal_us:9.2} {:7.0} {:7.0}  err max {:.2}x mean {:.3}x torch  {best_algo:?}",
                wbytes as f64 / torch_us / 1e3,
                wbytes as f64 / best_us / 1e3,
                best_err.0 / torch_err.0,
                best_err.1 / torch_err.1,
            );
            if best_us < torch_us * 0.97 {
                table.push(format!(
                    "    pick({rows}, {n_in}, {n_out}, [{}, {}, {}, {}, {}, {}, {}, {}, {}]), // {torch_us:.2} -> {best_us:.2} us",
                    best_algo.id, best_algo.tile, best_algo.stages, best_algo.splitk, best_algo.reduction,
                    best_algo.swizzle, best_algo.custom, best_algo.inner, best_algo.cluster,
                ));
            }
            report.push(json!({
                "shape": name, "n_in": n_in, "n_out": n_out, "rows": rows,
                "torch_us": torch_us, "copy_us": copy_us, "ideal_us": ideal_us,
                "best_us": best_us, "best": algo_json(&best_algo), "candidates": rows_json,
                "torch_max_err": torch_err.0, "torch_mean_err": torch_err.1,
            }));
        }
    }

    println!(
        "\ntable entries (best under 0.97x torch):\n{}",
        table.join("\n")
    );
    std::fs::write(
        &out,
        serde_json::to_vec_pretty(&json!({"cases": report, "table": table}))?,
    )?;
    ensure!(!report.is_empty(), "no cases");
    Ok(())
}
