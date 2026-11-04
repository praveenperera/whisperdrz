//! The cuda-oxide small-M linears (`cuda::small_m`) against an exact f64 product, under CUDA
//! graph capture and replay, and the probe that times them against the decode step's
//! current picks
//!
//! The accuracy test runs every decode linear shape at every graph bucket the kernels serve
//! (1 to 16 rows) on every configuration of the built kernels, the routed ones included. It
//! NaN-prefills the output (rows past `rows` too, which must stay NaN), asserts lengths,
//! checks two runs agree bit for bit, and bounds the error by torch's own call's against
//! the same exact product: within 1.25x its max-abs and 1.05x its mean-abs error, the
//! cuBLASLt decode table's bar. The fused GELU must equal `gelu_f16x8` run on the unfused
//! output, bit for bit. The graph test captures every routed cell on one shared workspace,
//! replays it on new inputs and compares each replay with direct launches bit for bit. The
//! probe (`WHISPERDRZ_SMALLM_PROBE` names its JSON report) times each cell's current pick,
//! the fastest non-split-K cuBLASLt algorithm the heuristic offers and every kernel
//! configuration, with weights rotated through a pool past the 96 MiB L2 as the decode step
//! streams them. GPU only: run them through the queue
#![cfg(feature = "cuda")]

#[allow(dead_code)]
#[path = "support/numerics.rs"]
mod numerics;

use std::path::PathBuf;

use cudarc::driver::{result as cur, sys as cu};
use eyre::{Result, ensure};
use half::f16;
use numerics::Rng;
use serde_json::json;
use whisperdrz_native::cuda::small_m::{self, SmallCfg, SmallM, SmallStrides};
use whisperdrz_native::cuda::{Blas, Gpu, ops};

/// Decode-step linears: name, depth, features, GELU after
const SHAPES: [(&str, usize, usize, bool); 4] = [
    ("qkv", 1280, 3840, false),
    ("proj", 1280, 1280, false),
    ("fc1", 1280, 5120, true),
    ("fc2", 5120, 1280, false),
];
/// The graph buckets the kernels serve
const ROWS: [usize; 5] = [1, 2, 4, 8, 16];
/// The built kernels, `(r16, ch)`
const KERNELS: [(bool, usize); 2] = [(false, 5), (true, 10)];
const NAN: u16 = 0x7e00;
const POOL: usize = 256 << 20;
const LAUNCHES: usize = 64;
const TRIALS: usize = 7;

/// The exact `y = x w^T + b` in f64, `[rows, n]`
fn truth(x: &[u16], w: &[u16], b: &[u16], (k, n): (usize, usize), rows: usize) -> Vec<f64> {
    let h = |v: u16| f16::from_bits(v).to_f64();
    let wf: Vec<f64> = w[..n * k].iter().map(|&v| h(v)).collect();
    let mut out = vec![0f64; rows * n];
    std::thread::scope(|scope| {
        for (r, row) in out.chunks_mut(n).enumerate() {
            let wf = &wf;
            scope.spawn(move || {
                let xr: Vec<f64> = (0..k).map(|i| h(x[r * k + i])).collect();
                for (f, y) in row.iter_mut().enumerate() {
                    let wr = &wf[f * k..(f + 1) * k];
                    *y = xr.iter().zip(wr).map(|(a, b)| a * b).sum::<f64>() + h(b[f]);
                }
            });
        }
    });
    out
}

/// Max-abs and mean-abs error of FP16 outputs against the exact values, failing on NaN
fn error(got: &[u16], truth: &[f64]) -> Result<(f64, f64)> {
    ensure!(
        got.len() == truth.len() && !got.is_empty(),
        "{} values against {}",
        got.len(),
        truth.len()
    );
    let (mut max, mut sum) = (0f64, 0f64);
    for (i, (g, t)) in got.iter().zip(truth).enumerate() {
        let g = f16::from_bits(*g).to_f64();
        ensure!(g.is_finite(), "non-finite output at {i}: {g}");
        let d = (g - t).abs();
        max = max.max(d);
        sum += d;
    }
    Ok((max, sum / got.len() as f64))
}

/// Every configuration of the built kernels for `rows` rows of a `k -> n` linear, up to 40
/// depth splits
fn configs((k, n): (usize, usize), rows: usize) -> Vec<SmallCfg> {
    let mut out = Vec::new();
    for (r16, ch) in KERNELS {
        for wk in [1, 2, 4] {
            let cfg = SmallCfg {
                r16,
                ch,
                wk,
                splits: 1,
            };
            if rows > cfg.max_rows()
                || !k.is_multiple_of(cfg.depth())
                || !n.is_multiple_of(cfg.features())
            {
                continue;
            }
            let cfg = SmallCfg {
                splits: k / cfg.depth(),
                ..cfg
            };
            if cfg.splits <= 40 {
                out.push(cfg);
            }
        }
    }
    out
}

/// Workspace for every configuration [`configs`] offers
fn workspace(gpu: &Gpu) -> Result<SmallM> {
    let mut part = 0;
    let mut counters = 0;
    for (_, k, n, _) in SHAPES {
        for rows in ROWS {
            for c in configs((k, n), rows) {
                part = part.max(c.part_floats(n, 1));
                counters = counters.max(c.counter_words(n, 1));
            }
        }
    }
    SmallM::new(gpu, part, counters)
}

#[test]
fn small_m_linears_within_f64_bound() -> Result<()> {
    if !numerics::gpu_enabled() {
        return Ok(());
    }
    let gpu = Gpu::new(0)?;
    let blas = Blas::new(&gpu)?;
    let small = workspace(&gpu)?;
    let mut rng = Rng::new(11);
    let max = 5120;
    let xh = rng.f16_normal(16 * max, 0.0, 1.0);
    let wh = rng.f16_normal(max * max, 0.0, 0.02);
    let bh = rng.f16_normal(max, 0.0, 0.1);
    let (x, w, b) = (
        gpu.upload(xh.clone())?,
        gpu.upload(wh.clone())?,
        gpu.upload(bh.clone())?,
    );
    let y = gpu.alloc(16 * max * 2)?;
    let mut worst = Vec::new();
    let mut routed = 0;
    for (name, k, n, gelu) in SHAPES {
        let exact = truth(&xh, &wh, &bh, (k, n), 16);
        for rows in ROWS {
            let len = rows * n;
            let want = &exact[..len];
            // torch's own call on the same operands, the bar
            gpu.upload_to(y.ptr(), bytemuck::cast_slice(&vec![NAN; 16 * n]))?;
            blas.linear_bias(x.ptr(), w.ptr(), b.ptr(), y.ptr(), (rows, k, n))?;
            let torch: Vec<u16> = gpu.download(&y)?;
            let torch_err = error(&torch[..len], want)?;

            let cfgs = configs((k, n), rows);
            if let Some(r) = small_m::route((rows, k, n)) {
                ensure!(
                    cfgs.contains(&r),
                    "{name} {rows}: routed {r:?} not among the tested configurations"
                );
                routed += 1;
            }
            for cfg in cfgs {
                let run = |gelu: bool| -> Result<Vec<u16>> {
                    gpu.upload_to(y.ptr(), bytemuck::cast_slice(&vec![NAN; 16 * n]))?;
                    let ops = (x.ptr(), w.ptr(), b.ptr(), y.ptr());
                    small.linear(
                        &gpu,
                        cfg,
                        ops,
                        (rows, k, n),
                        SmallStrides::dense(k, n),
                        gelu,
                    )?;
                    let got: Vec<u16> = gpu.download(&y)?;
                    ensure!(
                        got.len() >= 16 * n,
                        "{name} {rows}: output length {}",
                        got.len()
                    );
                    ensure!(
                        got[len..16 * n].iter().all(|&v| v == NAN),
                        "{name} {rows} {cfg:?}: wrote past row {rows}"
                    );
                    Ok(got[..len].to_vec())
                };
                let first = run(false)?;
                ensure!(
                    run(false)? == first,
                    "{name} {rows} {cfg:?}: two runs differ"
                );
                let err = error(&first, want)?;
                let ratio = (err.0 / torch_err.0, err.1 / torch_err.1);
                worst.push((ratio, name, rows, cfg));
                ensure!(
                    ratio.0 <= 1.25 && ratio.1 <= 1.05,
                    "{name} {rows} {cfg:?}: error {err:?} against torch's {torch_err:?}"
                );
                if gelu {
                    let fused = run(true)?;
                    let sep = gpu.upload(first.clone())?;
                    ops::gelu_inplace(&gpu, sep.ptr(), len)?;
                    let sep: Vec<u16> = gpu.download(&sep)?;
                    ensure!(
                        fused == sep[..len],
                        "{name} {rows} {cfg:?}: fused GELU differs from gelu_f16x8"
                    );
                }
            }
        }
    }
    worst.sort_by(|a, b| b.0.0.total_cmp(&a.0.0).then(b.0.1.total_cmp(&a.0.1)));
    for (r, name, rows, cfg) in worst.iter().take(8) {
        eprintln!(
            "worst max ratio {:.3} mean {:.3}: {name} {rows} {cfg:?}",
            r.0, r.1
        );
    }
    ensure!(worst.len() >= 40, "only {} cases ran", worst.len());
    ensure!(routed == 8, "{routed} routed cells tested, the table has 8");
    Ok(())
}

/// One captured CUDA graph and its executable, destroyed on drop
struct Graph {
    graph: cu::CUgraph,
    exec: cu::CUgraphExec,
}

impl Graph {
    /// Captures the launches `enqueue` makes on `gpu`'s stream
    fn capture(gpu: &Gpu, enqueue: impl FnOnce() -> Result<()>) -> Result<Self> {
        let stream = gpu.cu_stream();
        unsafe {
            cur::stream::begin_capture(
                stream,
                cu::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )?
        };
        let enqueued = enqueue();
        // end the capture whatever happened, so the stream is usable again
        let graph = unsafe { cur::stream::end_capture(stream)? };
        enqueued?;
        ensure!(!graph.is_null(), "empty capture");
        let flags = cu::CUgraphInstantiate_flags(0);
        let exec = match unsafe { cur::graph::instantiate(graph, flags) } {
            Ok(exec) => exec,
            Err(e) => {
                unsafe { cur::graph::destroy(graph)? };
                return Err(e.into());
            }
        };
        Ok(Self { graph, exec })
    }

    fn launch(&self, gpu: &Gpu) -> Result<()> {
        unsafe { cur::graph::launch(self.exec, gpu.cu_stream())? };
        Ok(())
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            let _ = cur::graph::exec_destroy(self.exec);
            let _ = cur::graph::destroy(self.graph);
        }
    }
}

#[test]
fn small_m_routes_replay_in_a_cuda_graph() -> Result<()> {
    if !numerics::gpu_enabled() {
        return Ok(());
    }
    let gpu = Gpu::new(0)?;
    // one workspace for every routed cell, as a decoder workspace holds it
    let small = SmallM::for_routes(&gpu)?;
    let mut rng = Rng::new(5);
    let max = 5120;
    let w = gpu.upload(rng.f16_normal(max * max, 0.0, 0.02))?;
    let b = gpu.upload(rng.f16_normal(max, 0.0, 0.1))?;
    let x = gpu.alloc(16 * max * 2)?;
    let mut cells = Vec::new();
    for (name, k, n, gelu) in SHAPES {
        for rows in ROWS {
            if let Some(cfg) = small_m::route((rows, k, n)) {
                cells.push((name, rows, k, n, gelu, cfg, gpu.alloc(rows * n * 2)?));
            }
        }
    }
    ensure!(cells.len() == 8, "{} routed cells", cells.len());
    let enqueue = || -> Result<()> {
        for (_, rows, k, n, gelu, cfg, y) in &cells {
            let ops = (x.ptr(), w.ptr(), b.ptr(), y.ptr());
            small.linear(
                &gpu,
                *cfg,
                ops,
                (*rows, *k, *n),
                SmallStrides::dense(*k, *n),
                *gelu,
            )?;
        }
        Ok(())
    };
    let outputs =
        || -> Result<Vec<Vec<u16>>> { cells.iter().map(|c| gpu.download(&c.6)).collect() };
    let prefill = || -> Result<()> {
        for c in &cells {
            gpu.upload_to(c.6.ptr(), bytemuck::cast_slice(&vec![NAN; c.1 * c.3]))?;
        }
        Ok(())
    };

    // load every kernel before the capture, as the decoder's warm-up does
    gpu.upload_to(
        x.ptr(),
        bytemuck::cast_slice(&rng.f16_normal(16 * max, 0.0, 1.0)),
    )?;
    enqueue()?;
    gpu.synchronize()?;
    let graph = Graph::capture(&gpu, enqueue)?;

    let mut last: Option<Vec<Vec<u16>>> = None;
    for replay in 0..4 {
        gpu.upload_to(
            x.ptr(),
            bytemuck::cast_slice(&rng.f16_normal(16 * max, 0.0, 1.0)),
        )?;
        prefill()?;
        graph.launch(&gpu)?;
        gpu.synchronize()?;
        let replayed = outputs()?;
        prefill()?;
        enqueue()?;
        let direct = outputs()?;
        for ((g, d), c) in replayed.iter().zip(&direct).zip(&cells) {
            let (name, rows) = (c.0, c.1);
            ensure!(
                g.len() == rows * c.3 && d.len() == g.len(),
                "{name} {rows}: output length {}",
                g.len()
            );
            ensure!(
                g.iter().all(|&v| f16::from_bits(v).is_finite()),
                "{name} {rows}: replay {replay} left a non-finite output"
            );
            ensure!(
                g == d,
                "{name} {rows}: replay {replay} differs from direct launches"
            );
        }
        if let Some(prev) = &last {
            ensure!(
                prev != &replayed,
                "replay {replay} gave the previous input's outputs"
            );
        }
        last = Some(replayed);
    }
    eprintln!(
        "{} routed cells replayed 4 times, bit-equal to direct launches",
        cells.len()
    );
    Ok(())
}

struct Bench<'a> {
    gpu: &'a Gpu,
    busy: (u64, u64),
}

impl Bench<'_> {
    /// Median over trials of the GPU time per launch of `launch(i)`, queued behind a long
    /// copy so the GPU's time is measured rather than the host's launch rate
    fn time_us(&self, mut launch: impl FnMut(usize) -> Result<()>) -> Result<f64> {
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

#[test]
fn probe_small_m() -> Result<()> {
    let Some(out) = std::env::var_os("WHISPERDRZ_SMALLM_PROBE").map(PathBuf::from) else {
        eprintln!("skipping: WHISPERDRZ_SMALLM_PROBE not set");
        return Ok(());
    };
    let gpu = Gpu::new(0)?;
    let blas = Blas::new(&gpu)?;
    let small = workspace(&gpu)?;
    let mut rng = Rng::new(7);
    let block = 8 << 20;
    let seed = gpu.upload(rng.f16_normal(block, 0.0, 0.02))?;
    let pool = gpu.alloc(POOL)?;
    for off in (0..POOL).step_by(block * 2) {
        gpu.copy_dtod(seed.ptr(), pool.at(off), block * 2)?;
    }
    let x = gpu.upload(rng.f16_normal(16 * 5120, 0.0, 1.0))?;
    let bias = gpu.upload(rng.f16_normal(5120, 0.0, 0.1))?;
    let y = gpu.alloc(16 * 5120 * 2)?;
    let busy = (gpu.alloc(POOL)?, gpu.alloc(POOL)?);
    let bench = Bench {
        gpu: &gpu,
        busy: (busy.0.ptr(), busy.1.ptr()),
    };
    // `rows` and `cur` or `r16,ch,wk,splits` (r16 0 or 1): one cell and one kernel, for ncu
    let one: Option<(usize, Option<SmallCfg>)> =
        std::env::var("WHISPERDRZ_SMALLM_ONE").ok().map(|v| {
            let (rows, what) = v
                .split_once(':')
                .expect("rows:cur or rows:r16,ch,wk,splits");
            let cfg = (what != "cur").then(|| {
                let p: Vec<usize> = what.split(',').map(|x| x.parse().unwrap()).collect();
                SmallCfg {
                    r16: p[0] == 1,
                    ch: p[1],
                    wk: p[2],
                    splits: p[3],
                }
            });
            (rows.parse().unwrap(), cfg)
        });
    let only = std::env::var("WHISPERDRZ_SMALLM_SHAPES").ok();

    let mut report = Vec::new();
    println!(
        "{:5} {:>4} {:>8} {:>8} {:>8} {:>7} {:>6} {:>6}  best",
        "shape", "rows", "cur us", "lt1 us", "best us", "speedup", "GB/s", "%1.67"
    );
    for (name, k, n, gelu) in SHAPES {
        if only
            .as_ref()
            .is_some_and(|o| !o.split(',').any(|s| s == name))
        {
            continue;
        }
        let wbytes = n * k * 2;
        let copies = POOL / wbytes;
        let w_at = |i: usize| pool.at((i % copies) * wbytes);
        let dims = (k, n);
        for rows in ROWS {
            if one.is_some_and(|(r, _)| r != rows) {
                continue;
            }
            let gelu_after = |blas_ok: Result<()>| -> Result<()> {
                blas_ok?;
                if gelu {
                    ops::gelu_inplace(&gpu, y.ptr(), rows * n)?;
                }
                Ok(())
            };
            let current = |i: usize| {
                gelu_after(blas.linear_bias_decode(
                    x.ptr(),
                    w_at(i),
                    bias.ptr(),
                    y.ptr(),
                    (rows, k, n),
                ))
            };
            if let Some((_, None)) = one {
                bench.time_us(current)?;
                continue;
            }
            let (cur_us, lt1_us) = if one.is_some() {
                (f64::NAN, f64::NAN)
            } else {
                // the fastest non-split-K algorithm the heuristic offers: split-K ones round
                // as if the product were FP16 before the bias below 32 rows
                let mut lt1 = f64::INFINITY;
                for c in
                    blas.linear_bias_candidates((x.ptr(), w_at(0), y.ptr()), (rows, k, n), 64)?
                {
                    if c.algo.splitk > 1 {
                        continue;
                    }
                    let run = |i: usize| {
                        gelu_after(blas.linear_bias_with(
                            x.ptr(),
                            w_at(i),
                            bias.ptr(),
                            y.ptr(),
                            (rows, k, n),
                            c.algo,
                        ))
                    };
                    if run(0).is_ok() {
                        lt1 = lt1.min(bench.time_us(run)?);
                    }
                }
                (bench.time_us(current)?, lt1)
            };
            let cfgs = match one {
                Some((_, Some(cfg))) => vec![cfg],
                _ => configs(dims, rows),
            };
            let mut cands = Vec::new();
            for cfg in cfgs {
                let us = bench.time_us(|i| {
                    small.linear(
                        &gpu,
                        cfg,
                        (x.ptr(), w_at(i), bias.ptr(), y.ptr()),
                        (rows, k, n),
                        SmallStrides::dense(k, n),
                        gelu,
                    )
                })?;
                cands.push((us, cfg));
            }
            cands.sort_by(|a, b| a.0.total_cmp(&b.0));
            let (best_us, best) = cands[0];
            let routed = small_m::route((rows, k, n));
            let routed_us = routed.and_then(|r| cands.iter().find(|c| c.1 == r).map(|c| c.0));
            let gbs = wbytes as f64 / best_us / 1e3;
            println!(
                "{name:5} {rows:4} {cur_us:8.2} {lt1_us:8.2} {best_us:8.2} {:7.2} {gbs:6.0} {:6.1}  {best:?} routed {routed_us:?}",
                cur_us / best_us,
                gbs / 1670.0 * 100.0
            );
            let cfg_json =
                |c: &SmallCfg| json!({"r16": c.r16, "ch": c.ch, "wk": c.wk, "splits": c.splits});
            report.push(json!({
                "shape": name, "rows": rows, "k": k, "n": n, "weight_bytes": wbytes,
                "current_us": cur_us, "lt_nonsplit_us": lt1_us, "best_us": best_us, "best": cfg_json(&best),
                "routed": routed.as_ref().map(cfg_json), "routed_us": routed_us,
                "candidates": cands.iter().map(|(us, c)| json!({"us": us, "cfg": cfg_json(c)})).collect::<Vec<_>>(),
            }));
        }
    }
    std::fs::write(
        &out,
        serde_json::to_vec_pretty(&json!({ "cases": report }))?,
    )?;
    ensure!(!report.is_empty() || one.is_some(), "no cases");
    Ok(())
}
