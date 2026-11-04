//! Decoder attention kernels against an f64 CPU reference, flash-attn and an FP32 truth
//!
//! `rust/tools/attn_ref.py` writes the torch side (`flash_attn_with_kvcache` in FP16 and
//! an FP32 attention on the same FP16 inputs); `rust/scripts/gpu-test.sh` runs both
//! through the GPU queue. The torch-backed tests skip when `WHISPERDRZ_ATTN_REF` doesn't
//! name its output.
//!
//! - synthetic cases (self-attention and cross-attention over the K/V caches): our output against an f64 attention on the exact FP16 inputs (at
//!   most one FP16 ulp, plus FP32-level slack near zero), ours and flash-attn's against the
//!   FP32 truth (ours no worse than 1.5x flash-attn's, max and mean), the appended cache
//!   positions bit for bit, every other cache position untouched, and the lengths after
//!   `advance_lens`. Cache positions past each slot's length are NaN, so a read past the
//!   length shows up as a NaN output
//! - trace: every decoder layer's self- and cross-attention on the inputs torch saw in a
//!   recorded prefill, with Q/K/V from the checkpoint through `Blas`, against torch's
//!   output and the FP32 truth per layer, and the layer-0 cross K/V cache bit for bit
//!   against torch's. The K/V path must stay within 1.5x flash-attn's error. The absorbed
//!   cross-attention of large decode buckets also runs on each layer's inputs and its
//!   max-abs and mean-abs errors are printed as ratios to flash-attn's, without a bar:
//!   folding the key projection into the query rounds `q W_k` to FP16 where torch rounds
//!   `xa W_k`, which puts it at 1.3-1.6x on mean-abs; the end-to-end gate judges it
//! - `absorbed_cross_within_f64_bound` (`WHISPERDRZ_GPU=1`, no torch side): the absorbed
//!   path's attention `O' = softmax(Q' xa^T) xa` on random encoder outputs, on several
//!   frame splits, against f64 within a bound derived from the kernel's own roundings, with
//!   the trash slot's encoder output NaN and every output NaN-prefilled; the default plan
//!   must repeat bit for bit and give the real rows the same bits whatever the trash slot
//!   holds
//!
//! `cross_bench` (`WHISPERDRZ_CROSS_BENCH=1`) times both cross-attention paths per layer
//! at each decode bucket, for the table behind `ABSORBED_MIN_ROWS`, and asserts nothing
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use eyre::{Result, bail, ensure, eyre};
use half::f16;
use serde_json::Value;
use whisperdrz_fixtures::{ArrayData, read_npy};
use whisperdrz_native::cuda::decoder_attn::{
    ABSORBED_MAX_SPLITS, AbsorbedPlan, Batch, CROSS_LEN, CrossKv, F16_NAN, HEAD_DIM, HEADS, Rows,
    SCALE, SELF_CAP, STATE, XA_BYTES,
};
use whisperdrz_native::cuda::{Blas, Buf, DecoderAttn, Gpu, KvCaches, Weights, nvtx};

#[allow(dead_code)]
#[path = "support/numerics.rs"]
mod numerics;

use numerics::Rng;

/// Ours against the FP32 truth may be at most this many times flash-attn's error
const TRUTH_RATIO: f64 = 1.5;
/// Absolute slack on top of one FP16 ulp against the f64 reference: FP32 rounding of the
/// softmax sums, which only matters for outputs near zero, where FP16 ulps are tiny
const F32_SLACK: f64 = 1.0 / (1 << 22) as f64;

struct RefDir(PathBuf);

impl RefDir {
    fn path(&self, file: &Value) -> Result<PathBuf> {
        let f = file
            .as_str()
            .ok_or_else(|| eyre!("bad file entry {file}"))?;
        Ok(self.0.join(f))
    }

    fn f16(&self, file: &Value) -> Result<(Vec<usize>, Vec<u16>)> {
        let a = read_npy(&self.path(file)?)?;
        let ArrayData::F16(v) = a.data else {
            bail!("{file} is not float16")
        };
        Ok((a.shape, v.iter().map(|h| h.to_bits()).collect()))
    }

    fn f32(&self, file: &Value) -> Result<(Vec<usize>, Vec<f32>)> {
        let a = read_npy(&self.path(file)?)?;
        let ArrayData::F32(v) = a.data else {
            bail!("{file} is not float32")
        };
        Ok((a.shape, v))
    }

    fn i32(&self, file: &Value) -> Result<Vec<i32>> {
        let a = read_npy(&self.path(file)?)?;
        let ArrayData::I32(v) = a.data else {
            bail!("{file} is not int32")
        };
        Ok(v)
    }
}

fn h(bits: u16) -> f64 {
    f16::from_bits(bits).to_f64()
}

/// FP16 spacing at `x`: 2^-24 in the subnormal range
fn ulp16(x: f64) -> f64 {
    let a = x.abs();
    if a < 2f64.powi(-14) {
        2f64.powi(-24)
    } else {
        2f64.powi(a.log2().floor() as i32 - 10)
    }
}

/// Error of FP16 outputs against an f64 reference, in ulps and in absolute terms
#[derive(Default)]
struct UlpStats {
    max_ulp: f64,
    /// Outputs that aren't the correctly rounded reference
    misrounded: usize,
    /// Outputs past one ulp plus the FP32 slack
    over: usize,
    n: usize,
}

impl UlpStats {
    fn add(&mut self, got: u16, want: f64) {
        let g = h(got);
        let err = (g - want).abs();
        self.n += 1;
        if !g.is_finite() {
            self.max_ulp = f64::INFINITY;
            self.over += 1;
            return;
        }

        self.max_ulp = self.max_ulp.max(err / ulp16(want));
        if f16::from_f64(want).to_bits() != got {
            self.misrounded += 1;
        }
        if err > ulp16(want) + F32_SLACK {
            self.over += 1;
        }
    }
}

impl std::fmt::Display for UlpStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max {:.3} ulp, {} of {} not correctly rounded, {} over 1 ulp",
            self.max_ulp, self.misrounded, self.n, self.over
        )
    }
}

/// Absolute error against the FP32 truth
#[derive(Default, Clone, Copy)]
struct AbsStats {
    max: f64,
    sum: f64,
    n: usize,
}

impl AbsStats {
    fn add(&mut self, got: f64, want: f64) {
        let e = (got - want).abs();
        self.max = if e.is_nan() {
            f64::INFINITY
        } else {
            self.max.max(e)
        };
        self.sum += e;
        self.n += 1;
    }

    fn mean(&self) -> f64 {
        self.sum / self.n.max(1) as f64
    }

    /// Ours is within `TRUTH_RATIO` of `flash` on both max and mean
    fn within(&self, flash: &AbsStats) -> bool {
        self.n > 0
            && self.n == flash.n
            && self.max <= TRUTH_RATIO * flash.max
            && self.mean() <= TRUTH_RATIO * flash.mean()
    }
}

/// One query row and head of attention in f64: `key(j)` gives the K and V head rows of
/// key `j < n`
fn attend_f64<'a>(
    q: &[u16],
    n: usize,
    key: impl Fn(usize) -> (&'a [u16], &'a [u16]),
) -> [f64; HEAD_DIM] {
    let mut scores = Vec::with_capacity(n);
    for j in 0..n {
        let (k, _) = key(j);
        let s: f64 = q.iter().zip(k).map(|(&a, &b)| h(a) * h(b)).sum();
        scores.push(s * SCALE as f64);
    }

    let mx = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut out = [0f64; HEAD_DIM];
    let mut sum = 0.0;
    for (j, s) in scores.iter().enumerate() {
        let p = (s - mx).exp();
        sum += p;
        let (_, v) = key(j);
        for (o, &x) in out.iter_mut().zip(v) {
            *o += p * h(x);
        }
    }

    out.map(|o| o / sum)
}

/// Offset of head `h` of position `j` of slot `s` in a `[slots, cap, HEADS, HEAD_DIM]` cache
fn at(s: usize, j: usize, cap: usize, hd: usize) -> usize {
    (s * cap + j) * STATE + hd * HEAD_DIM
}

struct Case {
    name: String,
    is_self: bool,
    rows: usize,
    real: usize,
    q_len: usize,
    n_slots: usize,
    cap: usize,
    q: Vec<u16>,
    k_new: Vec<u16>,
    v_new: Vec<u16>,
    k_cache: Vec<u16>,
    v_cache: Vec<u16>,
    slots: Vec<i32>,
    lens: Vec<i32>,
    flash: Vec<u16>,
    truth: Vec<f32>,
}

impl Case {
    fn load(dir: &RefDir, c: &Value) -> Result<Self> {
        let files = &c["files"];
        let dim = |k: &str| {
            c[k].as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| eyre!("case without {k}"))
        };
        let name = c["name"].as_str().unwrap_or("?").to_owned();
        let is_self = match c["kind"].as_str() {
            Some("self") => true,
            Some("cross") => false,
            other => bail!("{name}: unknown kind {other:?}"),
        };
        let (rows, real, q_len, n_slots, cap) = (
            dim("rows")?,
            dim("real_rows")?,
            dim("q_len")?,
            dim("slots")?,
            dim("cap")?,
        );
        ensure!(
            cap == if is_self { SELF_CAP } else { CROSS_LEN },
            "{name}: cap {cap}"
        );

        let tokens = rows * q_len * STATE;
        let (_, q) = dir.f16(&files["q"])?;
        let (_, k_cache) = dir.f16(&files["k_cache"])?;
        let (_, v_cache) = dir.f16(&files["v_cache"])?;
        let (_, flash) = dir.f16(&files["flash"])?;
        let (_, truth) = dir.f32(&files["truth"])?;
        let slots = dir.i32(&files["slots"])?;
        let (k_new, v_new, lens) = if is_self {
            (
                dir.f16(&files["k_new"])?.1,
                dir.f16(&files["v_new"])?.1,
                dir.i32(&files["lens"])?,
            )
        } else {
            (Vec::new(), Vec::new(), vec![0; n_slots])
        };

        let cache = n_slots * cap * STATE;
        ensure!(
            q.len() == tokens
                && flash.len() == tokens
                && truth.len() == tokens
                && k_cache.len() == cache
                && v_cache.len() == cache
                && slots.len() == rows
                && lens.len() == n_slots
                && (!is_self || (k_new.len() == tokens && v_new.len() == tokens)),
            "{name}: array sizes"
        );
        ensure!(
            slots.iter().all(|&s| (s as usize) < n_slots)
                && slots[real..].iter().all(|&s| s as usize == n_slots - 1),
            "{name}: slots {slots:?}"
        );

        Ok(Self {
            name,
            is_self,
            rows,
            real,
            q_len,
            n_slots,
            cap,
            q,
            k_new,
            v_new,
            k_cache,
            v_cache,
            slots,
            lens,
            flash,
            truth,
        })
    }

    fn tokens(&self) -> usize {
        self.rows * self.q_len
    }

    /// Cached keys a row sees before its new ones
    fn len_of(&self, row: usize) -> usize {
        if self.is_self {
            self.lens[self.slots[row] as usize] as usize
        } else {
            self.cap
        }
    }

    /// The cache with every position at or past its slot's length set to NaN
    fn poisoned(&self, cache: &[u16]) -> Vec<u16> {
        let mut c = cache.to_vec();
        if self.is_self {
            for s in 0..self.n_slots {
                let from = at(s, self.lens[s] as usize, self.cap, 0);
                c[from..at(s + 1, 0, self.cap, 0)].fill(F16_NAN);
            }
        }

        c
    }

    /// f64 attention for query `t` of `row`, head `hd`
    fn reference(&self, row: usize, t: usize, hd: usize) -> [f64; HEAD_DIM] {
        let s = self.slots[row] as usize;
        let len = self.len_of(row);
        let qi = ((row * self.q_len + t) * HEADS + hd) * HEAD_DIM;
        let n = if self.is_self { len + t + 1 } else { self.cap };
        attend_f64(&self.q[qi..qi + HEAD_DIM], n, |j| {
            if j < len {
                let a = at(s, j, self.cap, hd);
                (
                    &self.k_cache[a..a + HEAD_DIM],
                    &self.v_cache[a..a + HEAD_DIM],
                )
            } else {
                let a = ((row * self.q_len + j - len) * HEADS + hd) * HEAD_DIM;
                (&self.k_new[a..a + HEAD_DIM], &self.v_new[a..a + HEAD_DIM])
            }
        })
    }
}

struct Outcome {
    out: Vec<u16>,
    /// Self-attention only: the layer's K and V caches and the lengths after the step
    caches: Option<(Vec<u16>, Vec<u16>, Vec<i32>)>,
}

/// Runs one case through `DecoderAttn`; self-attention reads Q/K/V out of one fused
/// `[tokens, 3 * STATE]` buffer, so the token stride is exercised
fn run(gpu: &Gpu, attn: &DecoderAttn, c: &Case) -> Result<Outcome> {
    let caches = KvCaches::new(gpu, 1, c.n_slots - 1)?;
    let (k_dst, v_dst) = if c.is_self {
        caches.self_slot(0, 0)?
    } else {
        caches.cross_slot(0, 0)?
    };
    gpu.upload_to(k_dst, bytemuck::cast_slice(&c.poisoned(&c.k_cache)))?;
    gpu.upload_to(v_dst, bytemuck::cast_slice(&c.poisoned(&c.v_cache)))?;
    gpu.upload_to(caches.lens().ptr(), bytemuck::cast_slice(&c.lens))?;

    let slot_buf = gpu.upload(c.slots.clone())?;
    let batch = Batch {
        slots: slot_buf.ptr(),
        rows: c.rows,
        q_len: c.q_len,
    };
    let out = gpu.upload(vec![F16_NAN; c.tokens() * STATE])?;

    let caches_after = if c.is_self {
        let mut qkv = vec![F16_NAN; c.tokens() * 3 * STATE];
        for i in 0..c.tokens() {
            let row = &mut qkv[i * 3 * STATE..(i + 1) * 3 * STATE];
            row[..STATE].copy_from_slice(&c.q[i * STATE..(i + 1) * STATE]);
            row[STATE..2 * STATE].copy_from_slice(&c.k_new[i * STATE..(i + 1) * STATE]);
            row[2 * STATE..].copy_from_slice(&c.v_new[i * STATE..(i + 1) * STATE]);
        }
        let qkv = gpu.upload(qkv)?;
        let rows = |part: usize| Rows {
            ptr: qkv.at(part * STATE * 2),
            stride: 3 * STATE,
        };
        attn.self_attn(
            gpu,
            &caches,
            0,
            batch,
            (rows(0), rows(1), rows(2)),
            out.ptr(),
        )?;
        caches.advance_lens(gpu, batch)?;

        let elems = c.n_slots * c.cap * STATE;
        let mut k = vec![0u16; elems];
        let mut v = vec![0u16; elems];
        download_at(gpu, k_dst, &mut k)?;
        download_at(gpu, v_dst, &mut v)?;
        Some((k, v, gpu.download::<i32>(caches.lens())?))
    } else {
        let q = gpu.upload(c.q.clone())?;
        attn.cross_attn(gpu, &caches, 0, batch, Rows::dense(q.ptr()), out.ptr())?;
        None
    };

    Ok(Outcome {
        out: gpu.download(&out)?,
        caches: caches_after,
    })
}

/// Copies device memory at an address into `host`, through a scratch buffer
fn download_at(gpu: &Gpu, src: u64, host: &mut [u16]) -> Result<()> {
    let tmp = gpu.alloc(host.len() * 2)?;
    gpu.copy_dtod(src, tmp.ptr(), host.len() * 2)?;
    gpu.download_into(&tmp, 0, host)
}

/// Checks the appended positions hold the new keys and values bit for bit, every other
/// position of a real slot is untouched, and the lengths moved by `q_len`; the trash
/// slot's contents are skipped (padding rows race on it, as in the reference)
fn check_caches(c: &Case, (k, v, lens): &(Vec<u16>, Vec<u16>, Vec<i32>)) -> Result<()> {
    let trash = c.n_slots - 1;
    let mut want_k = c.poisoned(&c.k_cache);
    let mut want_v = c.poisoned(&c.v_cache);
    let mut want_lens = c.lens.clone();
    for row in 0..c.real {
        let s = c.slots[row] as usize;
        for t in 0..c.q_len {
            let pos = c.lens[s] as usize + t;
            let src = (row * c.q_len + t) * STATE;
            let dst = at(s, pos, c.cap, 0);
            want_k[dst..dst + STATE].copy_from_slice(&c.k_new[src..src + STATE]);
            want_v[dst..dst + STATE].copy_from_slice(&c.v_new[src..src + STATE]);
        }
        want_lens[s] += c.q_len as i32;
    }
    want_lens[trash] = 0;

    let real = at(trash, 0, c.cap, 0);
    let diff = |a: &[u16], b: &[u16]| {
        a[..real]
            .iter()
            .zip(&b[..real])
            .filter(|(x, y)| x != y)
            .count()
    };
    let (dk, dv) = (diff(k, &want_k), diff(v, &want_v));
    ensure!(
        dk == 0 && dv == 0,
        "{}: {dk} K and {dv} V cache elements differ from the expected append",
        c.name
    );
    ensure!(
        *lens == want_lens,
        "{}: lengths {lens:?}, want {want_lens:?}",
        c.name
    );
    Ok(())
}

fn ref_dir() -> Option<RefDir> {
    let dir = std::env::var_os("WHISPERDRZ_ATTN_REF").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("skipping: WHISPERDRZ_ATTN_REF not set (run scripts/gpu-test.sh)");
    }

    dir.map(RefDir)
}

fn manifest(dir: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&std::fs::read(
        dir.join("cases.json"),
    )?)?)
}

#[test]
fn synthetic_cases_match_f64_and_flash() -> Result<()> {
    let Some(dir) = ref_dir() else { return Ok(()) };
    let manifest = manifest(&dir.0)?;
    let cases = manifest["cases"].as_array().cloned().unwrap_or_default();
    ensure!(
        cases.len() >= 9,
        "{} cases in {}",
        cases.len(),
        dir.0.display()
    );

    let gpu = Gpu::new(0)?;
    let attn = DecoderAttn::new(&gpu)?;
    let mut failed = Vec::new();
    for value in &cases {
        let c = Case::load(&dir, value)?;
        let plan = if c.is_self {
            attn.self_plan(c.rows, c.q_len)
        } else {
            attn.cross_plan(c.rows, c.q_len)
        };
        let got = run(&gpu, &attn, &c)?;

        let mut ours_f64 = UlpStats::default();
        let mut flash_f64 = UlpStats::default();
        let mut ours_truth = AbsStats::default();
        let mut flash_truth = AbsStats::default();
        for row in 0..c.real {
            for t in 0..c.q_len {
                for hd in 0..HEADS {
                    let want = c.reference(row, t, hd);
                    let base = ((row * c.q_len + t) * HEADS + hd) * HEAD_DIM;
                    for (d, &w) in want.iter().enumerate() {
                        let i = base + d;
                        ours_f64.add(got.out[i], w);
                        flash_f64.add(c.flash[i], w);
                        ours_truth.add(h(got.out[i]), c.truth[i] as f64);
                        flash_truth.add(h(c.flash[i]), c.truth[i] as f64);
                    }
                }
            }
        }
        let pad_finite = got.out[c.real * c.q_len * STATE..]
            .iter()
            .all(|&b| h(b).is_finite());

        println!(
            "{}: rows {} ({} real) q_len {} splits {} chunk {}\n  ours vs f64: {ours_f64}\n  flash vs f64: {flash_f64}\n  vs fp32 truth: ours max {:.3e} mean {:.3e}, flash max {:.3e} mean {:.3e}",
            c.name,
            c.rows,
            c.real,
            c.q_len,
            plan.splits,
            plan.chunk,
            ours_truth.max,
            ours_truth.mean(),
            flash_truth.max,
            flash_truth.mean()
        );

        let mut why = Vec::new();
        if ours_f64.over > 0 || ours_f64.n == 0 {
            why.push("f64".to_owned());
        }
        if !ours_truth.within(&flash_truth) {
            why.push("truth".to_owned());
        }
        if !pad_finite {
            why.push("padding rows not finite".to_owned());
        }
        if let Some(caches) = &got.caches
            && let Err(e) = check_caches(&c, caches)
        {
            why.push(e.to_string());
        }
        if !why.is_empty() {
            println!("  FAIL: {why:?}");
            failed.push(format!("{}: {why:?}", c.name));
        }
    }

    ensure!(
        failed.is_empty(),
        "{} of {} cases failed: {failed:?}",
        failed.len(),
        cases.len()
    );
    Ok(())
}

/// Device copy of an FP16 array
fn upload16(gpu: &Gpu, v: &[u16]) -> Result<Buf> {
    gpu.upload(v.to_vec())
}

#[test]
fn trace_prefill_layers_match_torch() -> Result<()> {
    let Some(dir) = ref_dir() else { return Ok(()) };
    let manifest = manifest(&dir.0)?;
    let tr = &manifest["trace"];
    ensure!(tr.is_object(), "no trace section in {}", dir.0.display());
    let Some(weights_path) = std::env::var_os("WHISPERDRZ_WEIGHTS").map(PathBuf::from) else {
        bail!("WHISPERDRZ_WEIGHTS not set while WHISPERDRZ_ATTN_REF is");
    };

    let dim = |k: &str| {
        tr[k]
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| eyre!("trace without {k}"))
    };
    let (rows, q_len, layers) = (dim("rows")?, dim("q_len")?, dim("layers")?);
    let trace_slots: Vec<usize> = tr["slots"]
        .as_array()
        .ok_or_else(|| eyre!("trace without slots"))?
        .iter()
        .map(|v| {
            v.as_u64()
                .map(|s| s as usize)
                .ok_or_else(|| eyre!("bad slot {v}"))
        })
        .collect::<Result<_>>()?;
    ensure!(
        trace_slots.len() == rows && layers == 32 && q_len == 7,
        "trace shape"
    );
    // windows on descending slots past 0..rows, so fill_cross takes its scratch-and-copy
    // path for every layer; layer 0 also fills 0..rows to cover the direct GEMM path
    let slot_list: Vec<usize> = (0..rows).map(|i| 2 * rows - 1 - i).collect();
    let direct_slots: Vec<usize> = (0..rows).collect();
    let tokens = rows * q_len;
    let files = &tr["files"];

    let gpu = Gpu::new(0)?;
    let blas = Blas::new(&gpu)?;
    let attn = DecoderAttn::new(&gpu)?;
    let weights = Weights::load(&gpu, &weights_path, |n| {
        n.starts_with("decoder.blocks.")
            && (n.contains(".attn.") || n.contains(".cross_attn."))
            && !n.contains(".out.")
    })?;
    let caches = KvCaches::new(&gpu, layers, 2 * rows)?;
    let slot_buf = gpu.upload(slot_list.iter().map(|&s| s as i32).collect::<Vec<i32>>())?;
    let batch = Batch {
        slots: slot_buf.ptr(),
        rows,
        q_len,
    };
    let scratch = gpu.alloc(rows * CROSS_LEN * STATE * 2)?;

    let (xa_shape, xa) = dir.f16(&files["xa"])?;
    ensure!(
        xa_shape == [rows, CROSS_LEN, STATE],
        "xa shape {xa_shape:?}"
    );
    let xa = upload16(&gpu, &xa)?;
    caches.store_xa(&gpu, xa.ptr(), &slot_list)?;

    let load = |key: &str| -> Result<(Vec<u16>, Vec<u16>, Vec<f32>)> {
        let (sx, x) = dir.f16(&files[format!("{key}_x")])?;
        let (sf, flash) = dir.f16(&files[format!("{key}_flash")])?;
        let (st, truth) = dir.f32(&files[format!("{key}_truth")])?;
        let want = [layers, rows, q_len, STATE];
        ensure!(
            sx == want && sf == want && st == want,
            "{key} shapes {sx:?} {sf:?} {st:?}"
        );
        Ok((x, flash, truth))
    };
    let (self_x, self_flash, self_truth) = load("self")?;
    let (cross_x, cross_flash, cross_truth) = load("cross")?;

    caches.reset_lens(&gpu, batch)?;
    let w = |layer: usize, name: &str, shape: &[usize]| -> Result<u64> {
        Ok(weights
            .get(&format!("decoder.blocks.{layer}.{name}"), shape)?
            .ptr())
    };
    let sq = [STATE, STATE];
    let per_layer = tokens * STATE;
    let x = gpu.alloc(per_layer * 2)?;
    let (q, k, v) = (
        gpu.alloc(per_layer * 2)?,
        gpu.alloc(per_layer * 2)?,
        gpu.alloc(per_layer * 2)?,
    );
    let out = gpu.alloc(per_layer * 2)?;
    // the absorbed cross-attention's per-head query and output, [tokens, HEADS, STATE]
    let (qk, ok) = (
        gpu.alloc(per_layer * HEADS * 2)?,
        gpu.alloc(per_layer * HEADS * 2)?,
    );
    let nan = vec![F16_NAN; per_layer];
    // cross f64 checks are the slow part, so they cover the first, a middle and the last layer
    let f64_cross_layers = [0, layers / 2, layers - 1];

    println!(
        "trace {} seq {}: {rows} windows x {q_len} tokens, torch slots {trace_slots:?}, ours {slot_list:?}",
        tr["stem"], tr["seq"]
    );
    println!(
        "layer | self ours max/mean | self flash max/mean | cross ours max/mean | cross flash max/mean | f64 self / cross | absorbed / flash max, mean (no bar)"
    );
    let mut failed = Vec::new();
    for layer in 0..layers {
        let range = layer * per_layer..(layer + 1) * per_layer;

        // self-attention on an empty cache: Q, K, V from the captured LayerNorm output
        gpu.upload_to(x.ptr(), bytemuck::cast_slice(&self_x[range.clone()]))?;
        blas.linear_bias(
            x.ptr(),
            w(layer, "attn.query.weight", &sq)?,
            w(layer, "attn.query.bias", &[STATE])?,
            q.ptr(),
            (tokens, STATE, STATE),
        )?;
        blas.linear(
            x.ptr(),
            w(layer, "attn.key.weight", &sq)?,
            k.ptr(),
            (tokens, STATE, STATE),
        )?;
        blas.linear_bias(
            x.ptr(),
            w(layer, "attn.value.weight", &sq)?,
            w(layer, "attn.value.bias", &[STATE])?,
            v.ptr(),
            (tokens, STATE, STATE),
        )?;
        gpu.upload_to(out.ptr(), bytemuck::cast_slice(&nan))?;
        attn.self_attn(
            &gpu,
            &caches,
            layer,
            batch,
            (
                Rows::dense(q.ptr()),
                Rows::dense(k.ptr()),
                Rows::dense(v.ptr()),
            ),
            out.ptr(),
        )?;
        let self_out: Vec<u16> = gpu.download(&out)?;
        let (sq_h, sk_h, sv_h): (Vec<u16>, Vec<u16>, Vec<u16>) =
            (gpu.download(&q)?, gpu.download(&k)?, gpu.download(&v)?);

        let mut self_ulp = UlpStats::default();
        for row in 0..rows {
            for t in 0..q_len {
                for hd in 0..HEADS {
                    let qi = (row * q_len + t) * STATE + hd * HEAD_DIM;
                    let want = attend_f64(&sq_h[qi..qi + HEAD_DIM], t + 1, |j| {
                        let a = (row * q_len + j) * STATE + hd * HEAD_DIM;
                        (&sk_h[a..a + HEAD_DIM], &sv_h[a..a + HEAD_DIM])
                    });
                    for (d, &wv) in want.iter().enumerate() {
                        self_ulp.add(self_out[qi + d], wv);
                    }
                }
            }
        }

        // cross-attention: K/V from the encoder output through the checkpoint
        let cross_w = (
            w(layer, "cross_attn.key.weight", &sq)?,
            w(layer, "cross_attn.value.weight", &sq)?,
            w(layer, "cross_attn.value.bias", &[STATE])?,
        );
        caches.fill_cross(
            &gpu,
            &blas,
            layer,
            xa.ptr(),
            cross_w,
            (&slot_list, Some(&scratch)),
        )?;
        if layer == 0 {
            caches.fill_cross(&gpu, &blas, 0, xa.ptr(), cross_w, (&direct_slots, None))?;
        }
        gpu.upload_to(x.ptr(), bytemuck::cast_slice(&cross_x[range.clone()]))?;
        blas.linear_bias(
            x.ptr(),
            w(layer, "cross_attn.query.weight", &sq)?,
            w(layer, "cross_attn.query.bias", &[STATE])?,
            q.ptr(),
            (tokens, STATE, STATE),
        )?;
        gpu.upload_to(out.ptr(), bytemuck::cast_slice(&nan))?;
        attn.cross_attn(&gpu, &caches, layer, batch, Rows::dense(q.ptr()), out.ptr())?;
        let cross_out: Vec<u16> = gpu.download(&out)?;

        // the absorbed path on the same query: each head's key projection, attention over
        // the encoder output, then each head's value projection
        let kv = CrossKv {
            wk: cross_w.0,
            wv: cross_w.1,
            bv: cross_w.2,
        };
        gpu.upload_to(out.ptr(), bytemuck::cast_slice(&nan))?;
        kv.query_keys(&blas, q.ptr(), qk.ptr(), tokens)?;
        attn.cross_absorbed(&gpu, &caches, batch, qk.ptr(), ok.ptr())?;
        kv.values(&blas, ok.ptr(), out.ptr(), tokens)?;
        let absorbed_out: Vec<u16> = gpu.download(&out)?;

        let mut cross_ulp = UlpStats::default();
        if f64_cross_layers.contains(&layer) {
            let cq: Vec<u16> = gpu.download(&q)?;
            let slot_elems = CROSS_LEN * STATE;
            let read = |slots: &[usize]| -> Result<(Vec<u16>, Vec<u16>)> {
                let mut ck = vec![0u16; rows * slot_elems];
                let mut cv = vec![0u16; rows * slot_elems];
                for (i, &s) in slots.iter().enumerate() {
                    let (kp, vp) = caches.cross_slot(layer, s)?;
                    download_at(&gpu, kp, &mut ck[i * slot_elems..(i + 1) * slot_elems])?;
                    download_at(&gpu, vp, &mut cv[i * slot_elems..(i + 1) * slot_elems])?;
                }
                Ok((ck, cv))
            };
            let (ck, cv) = read(&slot_list)?;

            if layer == 0 {
                let (_, tk) = dir.f16(&files["cross_k0"])?;
                let (_, tv) = dir.f16(&files["cross_v0"])?;
                for (path, (k, v)) in [
                    ("scratch", (ck.clone(), cv.clone())),
                    ("direct", read(&direct_slots)?),
                ] {
                    let dk = k.iter().zip(&tk).filter(|(a, b)| a != b).count();
                    let dv = v.iter().zip(&tv).filter(|(a, b)| a != b).count();
                    println!(
                        "layer 0 cross K/V cache ({path} path) vs torch: {dk} K and {dv} V of {} elements differ",
                        k.len()
                    );
                    if dk > 0 || dv > 0 || tk.len() != k.len() || tv.len() != v.len() {
                        failed.push(format!(
                            "layer 0 cross cache ({path}): {dk} K, {dv} V differ"
                        ));
                    }
                }
            }

            if f64_cross_layers.contains(&layer) {
                for row in 0..rows {
                    for t in 0..q_len {
                        for hd in 0..HEADS {
                            let qi = (row * q_len + t) * STATE + hd * HEAD_DIM;
                            let want = attend_f64(&cq[qi..qi + HEAD_DIM], CROSS_LEN, |j| {
                                let a = (row * CROSS_LEN + j) * STATE + hd * HEAD_DIM;
                                (&ck[a..a + HEAD_DIM], &cv[a..a + HEAD_DIM])
                            });
                            for (d, &wv) in want.iter().enumerate() {
                                cross_ulp.add(cross_out[qi + d], wv);
                            }
                        }
                    }
                }
            }
        }

        let score = |ours: &[u16], flash: &[u16], truth: &[f32]| {
            let (mut o, mut f) = (AbsStats::default(), AbsStats::default());
            for i in 0..per_layer {
                o.add(h(ours[i]), truth[range.start + i] as f64);
                f.add(h(flash[range.start + i]), truth[range.start + i] as f64);
            }
            (o, f)
        };
        let (so, sf) = score(&self_out, &self_flash, &self_truth);
        let (co, cf) = score(&cross_out, &cross_flash, &cross_truth);
        let (ca, _) = score(&absorbed_out, &cross_flash, &cross_truth);
        let cross_f64 = if cross_ulp.n > 0 {
            format!("{:.3}", cross_ulp.max_ulp)
        } else {
            "-".to_owned()
        };
        println!(
            "{layer:5} | {:.3e} / {:.3e} | {:.3e} / {:.3e} | {:.3e} / {:.3e} | {:.3e} / {:.3e} | {:.3} / {cross_f64} ulp | {:.2}x, {:.2}x",
            so.max,
            so.mean(),
            sf.max,
            sf.mean(),
            co.max,
            co.mean(),
            cf.max,
            cf.mean(),
            self_ulp.max_ulp,
            ca.max / cf.max,
            ca.mean() / cf.mean()
        );
        if !ca.max.is_finite() || !ca.mean().is_finite() {
            failed.push(format!("layer {layer} absorbed cross not finite"));
        }
        if !so.within(&sf) {
            failed.push(format!("layer {layer} self vs truth"));
        }
        if !co.within(&cf) {
            failed.push(format!("layer {layer} cross vs truth"));
        }
        if self_ulp.over > 0 || cross_ulp.over > 0 {
            failed.push(format!(
                "layer {layer} f64: self {self_ulp}; cross {cross_ulp}"
            ));
        }
    }

    caches.advance_lens(&gpu, batch)?;
    let lens: Vec<i32> = gpu.download(caches.lens())?;
    for (s, &l) in lens.iter().enumerate() {
        let want = if slot_list.contains(&s) {
            q_len as i32
        } else {
            0
        };
        if l != want {
            failed.push(format!("slot {s} length {l}, want {want}"));
        }
    }

    ensure!(failed.is_empty(), "trace checks failed: {failed:?}");
    Ok(())
}
/// Real slots of the absorbed cross-attention cases; rows share them, since
/// cross-attention only reads its slot
const XA_SLOTS: usize = 12;

/// A synthetic absorbed cross-attention case: `real` rows on random real slots, then `pad`
/// padding rows on the trash slot, `Q'` drawn with standard deviation `q_std` over unit
/// normal encoder outputs, so scores spread by `q_std * sqrt(STATE)`: 0.36 is nearly flat
/// over the 1500 frames, 3.6 like the trace, 10.7 a few frames
struct AbsorbedCase {
    name: &'static str,
    real: usize,
    pad: usize,
    q_len: usize,
    q_std: f64,
}

const ABSORBED_CASES: [AbsorbedCase; 5] = [
    AbsorbedCase {
        name: "decode_b52",
        real: 46,
        pad: 6,
        q_len: 1,
        q_std: 0.1,
    },
    AbsorbedCase {
        name: "decode_b8_peaked",
        real: 8,
        pad: 0,
        q_len: 1,
        q_std: 0.3,
    },
    AbsorbedCase {
        name: "decode_b2_flat",
        real: 2,
        pad: 0,
        q_len: 1,
        q_std: 0.01,
    },
    AbsorbedCase {
        name: "decode_b1",
        real: 1,
        pad: 0,
        q_len: 1,
        q_std: 0.1,
    },
    AbsorbedCase {
        name: "prefill_b5",
        real: 5,
        pad: 0,
        q_len: 7,
        q_std: 0.1,
    },
];

/// Each real slot's encoder output in f64 and, per dimension, the sum of its magnitudes
/// over the frames
struct XaHost {
    xa: Vec<Vec<f64>>,
    abs_sum: Vec<Vec<f64>>,
}

/// `softmax(q xa^T) xa` in f64 for one query row of `Q'`, and each output's error bound
/// for a kernel on these FP16 inputs that sums in FP32, rounds the softmax weights to FP16
/// and the output to FP16: the weights' rounding (2^-11 relative, and 2^-25 absolute per
/// frame in FP16's subnormal range, the row sum being at least 1), FP32 sums of up to 1500
/// terms and the scores' FP32 rounding inside 2^-12 relative, then half an FP16 ulp
fn absorbed_f64(q: &[u16], xa: &[f64], abs_sum: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let q: Vec<f64> = q.iter().map(|&b| h(b)).collect();
    let scores: Vec<f64> = xa
        .as_chunks::<STATE>()
        .0
        .iter()
        .map(|x| x.iter().zip(&q).map(|(a, b)| a * b).sum())
        .collect();
    let mx = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let (mut out, mut mag, mut l) = (vec![0f64; STATE], vec![0f64; STATE], 0f64);
    for (x, &s) in xa.as_chunks::<STATE>().0.iter().zip(&scores) {
        let p = (s - mx).exp();
        l += p;
        for ((o, m), &v) in out.iter_mut().zip(mag.iter_mut()).zip(x) {
            *o += p * v;
            *m += p * v.abs();
        }
    }

    let rel = 2f64.powi(-11) + 2f64.powi(-12);
    let bound = out
        .iter_mut()
        .zip(&mag)
        .zip(abs_sum)
        .map(|((o, &m), &a)| {
            *o /= l;
            let e = rel * m / l + 2f64.powi(-25) * a;
            e + 0.5 * ulp16(o.abs() + e)
        })
        .collect();
    (out, bound)
}

/// [`absorbed_f64`] for the first `tokens` tokens of `q` (`[tokens, HEADS, STATE]`), token
/// `t` reading the slot of row `t / q_len`, across the host's cores
fn absorbed_truth(
    host: &XaHost,
    q: &[u16],
    slots: &[usize],
    (tokens, q_len): (usize, usize),
) -> (Vec<f64>, Vec<f64>) {
    let n = tokens * HEADS;
    let mut want = vec![0f64; n * STATE];
    let mut bound = vec![0f64; n * STATE];
    let threads = std::thread::available_parallelism().map_or(8, usize::from);
    let per = n.div_ceil(threads);
    std::thread::scope(|scope| {
        for ((w, b), first) in want
            .chunks_mut(per * STATE)
            .zip(bound.chunks_mut(per * STATE))
            .zip((0..).step_by(per))
        {
            scope.spawn(move || {
                for (i, (wr, br)) in w
                    .as_chunks_mut::<STATE>()
                    .0
                    .iter_mut()
                    .zip(b.as_chunks_mut::<STATE>().0)
                    .enumerate()
                {
                    let qi = first + i;
                    let slot = slots[qi / HEADS / q_len];
                    let (o, e) = absorbed_f64(
                        &q[qi * STATE..(qi + 1) * STATE],
                        &host.xa[slot],
                        &host.abs_sum[slot],
                    );
                    wr.copy_from_slice(&o);
                    br.copy_from_slice(&e);
                }
            });
        }
    });
    (want, bound)
}

/// The worst error of `got` as a fraction of its bound; fails on a non-finite output or a
/// length mismatch
fn absorbed_check(got: &[u16], want: &[f64], bound: &[f64]) -> Result<f64> {
    ensure!(
        got.len() == want.len() && want.len() == bound.len() && !want.is_empty(),
        "{} outputs against {} references",
        got.len(),
        want.len()
    );
    let mut worst = 0f64;
    for (i, ((&g, &w), &b)) in got.iter().zip(want).zip(bound).enumerate() {
        let g = h(g);
        ensure!(g.is_finite(), "output {i} is {g}");
        worst = worst.max((g - w).abs() / b);
    }

    Ok(worst)
}

#[test]
fn absorbed_cross_within_f64_bound() -> Result<()> {
    if !numerics::gpu_enabled() {
        return Ok(());
    }

    let gpu = Gpu::new(0)?;
    let attn = DecoderAttn::new(&gpu)?;
    let caches = KvCaches::new(&gpu, 1, XA_SLOTS)?;
    let trash = caches.trash();
    let mut rng = Rng::new(7);
    let mut host = XaHost {
        xa: Vec::new(),
        abs_sum: Vec::new(),
    };
    for s in 0..XA_SLOTS {
        let bits = rng.f16_normal(CROSS_LEN * STATE, 0.0, 1.0);
        gpu.upload_to(caches.xa_slot(s)?, bytemuck::cast_slice(&bits))?;
        let xa: Vec<f64> = bits.iter().map(|&b| h(b)).collect();
        let mut abs_sum = vec![0f64; STATE];
        for x in xa.as_chunks::<STATE>().0 {
            for (a, &v) in abs_sum.iter_mut().zip(x) {
                *a += v.abs();
            }
        }
        host.xa.push(xa);
        host.abs_sum.push(abs_sum);
    }
    // padding rows read the trash slot; NaN there shows any leak into real rows
    let fill_trash = |bits: u16| -> Result<()> {
        gpu.upload_to(
            caches.xa_slot(trash)?,
            bytemuck::cast_slice(&vec![bits; CROSS_LEN * STATE]),
        )
    };
    fill_trash(F16_NAN)?;
    let max_parts = gpu.sms().max(ABSORBED_MAX_SPLITS);

    let mut failed = Vec::new();
    for c in &ABSORBED_CASES {
        let rows = c.real + c.pad;
        let (tokens, real_tokens) = (rows * c.q_len, c.real * c.q_len);
        let slots: Vec<usize> = (0..rows)
            .map(|r| {
                if r < c.real {
                    (rng.next_u64() % XA_SLOTS as u64) as usize
                } else {
                    trash
                }
            })
            .collect();
        let q = rng.f16_normal(tokens * HEADS * STATE, 0.0, c.q_std);
        let (want, bound) = absorbed_truth(&host, &q, &slots, (real_tokens, c.q_len));

        let slot_buf = gpu.upload(slots.iter().map(|&s| s as i32).collect::<Vec<i32>>())?;
        let batch = Batch {
            slots: slot_buf.ptr(),
            rows,
            q_len: c.q_len,
        };
        let n = tokens * HEADS * STATE;
        let real_n = real_tokens * HEADS * STATE;
        let q = gpu.upload(q)?;
        let out = gpu.alloc(n * 2)?;
        let run = |plan: AbsorbedPlan| -> Result<Vec<u16>> {
            gpu.upload_to(out.ptr(), bytemuck::cast_slice(&vec![F16_NAN; n]))?;
            attn.cross_absorbed_with(&gpu, &caches, batch, (q.ptr(), out.ptr()), plan)?;
            gpu.download(&out)
        };

        let default = attn.absorbed_plan(rows, c.q_len);
        let mut plans = vec![default];
        for splits in [1, 3, ABSORBED_MAX_SPLITS] {
            let p = AbsorbedPlan::with_splits(splits);
            if !plans.contains(&p) && p.splits * tokens <= max_parts {
                plans.push(p);
            }
        }
        let first = run(default)?;
        for &plan in &plans {
            let label = format!("{} splits of {} tiles", plan.splits, plan.tiles);
            let got = if plan == default {
                first.clone()
            } else {
                run(plan)?
            };
            match absorbed_check(&got[..real_n], &want, &bound) {
                Ok(worst) => {
                    println!(
                        "{} (rows {rows}, {} real, q_len {}) {label}: worst error {worst:.3} of the bound",
                        c.name, c.real, c.q_len
                    );
                    if worst > 1.0 {
                        failed.push(format!("{} {label}: {worst:.3} of the bound", c.name));
                    }
                }
                Err(e) => failed.push(format!("{} {label}: {e}", c.name)),
            }
        }

        if run(default)? != first {
            failed.push(format!("{}: the default plan isn't deterministic", c.name));
        }
        // zeros, what the model's never-written trash slot holds: the real rows keep their
        // bits and the padding rows come out finite
        fill_trash(0)?;
        let clean = run(default)?;
        fill_trash(F16_NAN)?;
        if clean[..real_n] != first[..real_n] {
            failed.push(format!("{}: real rows depend on the trash slot", c.name));
        }
        if !clean[real_n..].iter().all(|&b| h(b).is_finite()) {
            failed.push(format!("{}: padding rows not finite", c.name));
        }
    }

    ensure!(
        failed.is_empty(),
        "absorbed cross-attention failed: {failed:?}"
    );
    Ok(())
}

/// Bytes the bench clears out of L2 before each timed call (the 5090 has 96 MB)
const FLUSH_BYTES: usize = 256 << 20;

/// Per-layer GPU times of both cross-attention paths and their pieces at each decode
/// bucket and two prefill shapes, the table behind `ABSORBED_MIN_ROWS`; asserts nothing
///
/// Runs with `WHISPERDRZ_CROSS_BENCH=1`. Each timed call follows an L2 flush unless its
/// name ends in `warm`; `WHISPERDRZ_CROSS_BENCH_REPS` sets the repetitions (default 25)
/// and the median is printed. With `WHISPERDRZ_NVTX_LIB` set, each call sits in an NVTX
/// range `xb_<piece>_r<rows>q<q_len>` for `ncu --nvtx-include`
#[test]
fn cross_bench() -> Result<()> {
    if std::env::var_os("WHISPERDRZ_CROSS_BENCH").is_none_or(|v| v != "1") {
        eprintln!("skipping: WHISPERDRZ_CROSS_BENCH not set");
        return Ok(());
    }
    let reps: usize = match std::env::var("WHISPERDRZ_CROSS_BENCH_REPS") {
        Ok(v) => v.parse()?,
        Err(_) => 25,
    };

    let gpu = Gpu::new(0)?;
    let blas = Blas::new(&gpu)?;
    let attn = DecoderAttn::new(&gpu)?;
    let max_rows = 54;
    // one layer: its zero K/V caches time the same as filled ones
    let caches = KvCaches::new(&gpu, 1, max_rows)?;
    let mut rng = Rng::new(11);
    let xa0 = gpu.upload(rng.f16_normal(CROSS_LEN * STATE, 0.0, 1.0))?;
    for s in 0..max_rows {
        gpu.copy_dtod(xa0.ptr(), caches.xa_slot(s)?, XA_BYTES)?;
    }
    let wk = gpu.upload(rng.f16_normal(STATE * STATE, 0.0, 0.03))?;
    let wv = gpu.upload(rng.f16_normal(STATE * STATE, 0.0, 0.03))?;
    let bv = gpu.upload(rng.f16_normal(STATE, 0.0, 0.1))?;
    let kv = CrossKv {
        wk: wk.ptr(),
        wv: wv.ptr(),
        bv: bv.ptr(),
    };
    let flush = gpu.alloc(FLUSH_BYTES)?;
    let max_parts = gpu.sms().max(ABSORBED_MAX_SPLITS);

    let shapes = [
        (1, 1),
        (2, 1),
        (4, 1),
        (8, 1),
        (16, 1),
        (32, 1),
        (42, 1),
        (52, 1),
        (54, 1),
        (8, 7),
        (52, 7),
    ];
    for (rows, q_len) in shapes {
        let tokens = rows * q_len;
        let slot_buf = gpu.upload((0..rows as i32).collect::<Vec<i32>>())?;
        let batch = Batch {
            slots: slot_buf.ptr(),
            rows,
            q_len,
        };
        let q = gpu.upload(rng.f16_normal(tokens * STATE, 0.0, 1.0))?;
        let qk = gpu.upload(rng.f16_normal(tokens * HEADS * STATE, 0.0, 0.1))?;
        let ok = gpu.alloc(tokens * HEADS * STATE * 2)?;
        let out = gpu.alloc(tokens * STATE * 2)?;

        let time = |name: &str, f: &dyn Fn() -> Result<()>| -> Result<()> {
            // the first call builds cuBLASLt plans
            f()?;
            let warm = name.ends_with("warm");
            let mut us = Vec::with_capacity(reps);
            for _ in 0..reps {
                if !warm {
                    gpu.memset_zero(flush.ptr(), FLUSH_BYTES)?;
                }
                let ms = gpu.time_ms(|| {
                    let _range = nvtx::Range::new(&format!("xb_{name}_r{rows}q{q_len}"));
                    f()
                })?;
                us.push(ms * 1000.0);
            }
            us.sort_by(f32::total_cmp);
            let med = us[us.len() / 2];
            println!(
                "xbench rows {rows:2} q_len {q_len} {name:>14}: median {med:8.1} us, min {:8.1} us, xa at {:6.0} GB/s",
                us[0],
                (rows * XA_BYTES) as f64 / (f64::from(med) * 1e3)
            );
            Ok(())
        };

        // the K/V path: attn_decode_f16 over the cached keys and values, then the combine
        let kv_path = || attn.cross_attn(&gpu, &caches, 0, batch, Rows::dense(q.ptr()), out.ptr());
        time("kv", &kv_path)?;
        time("kv_warm", &kv_path)?;

        // the absorbed path: both projections around the fused kernel
        time("qk", &|| kv.query_keys(&blas, q.ptr(), qk.ptr(), tokens))?;
        time("values", &|| kv.values(&blas, ok.ptr(), out.ptr(), tokens))?;
        let default = attn.absorbed_plan(rows, q_len);
        let fused = |p: AbsorbedPlan| {
            attn.cross_absorbed_with(&gpu, &caches, batch, (qk.ptr(), ok.ptr()), p)
        };
        time(&format!("fused_d{}", default.splits), &|| fused(default))?;
        time("fused_warm", &|| fused(default))?;
        time("absorbed", &|| {
            kv.query_keys(&blas, q.ptr(), qk.ptr(), tokens)?;
            fused(default)?;
            kv.values(&blas, ok.ptr(), out.ptr(), tokens)
        })?;
        let mut seen = vec![default];
        for splits in [1, 2, 3, 4, 6, 8, 16, 32, 94] {
            let p = AbsorbedPlan::with_splits(splits);
            if seen.contains(&p) || p.splits * tokens > max_parts {
                continue;
            }
            seen.push(p);
            time(&format!("fused_s{}", p.splits), &|| fused(p))?;
        }
    }

    Ok(())
}
