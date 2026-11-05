//! Speculative verify steps on the GPU: the ragged attention kernels against f64 and the
//! one-token step's attention, and the verify forward against torch and the step
//!
//! - `verify_kernels_match_f64` (`WHISPERDRZ_GPU=1`): `attn_verify_q{2,4,8}_f16` on
//!   ragged rows (1 to 8 queries, distinct slots out of order, cache lengths from 0 up to
//!   the cap, padding rows on the trash slot), self- and cross-attention, every output
//!   within one FP16 ulp (plus FP32 slack near zero) of an f64 attention on the exact FP16
//!   inputs. Outputs are NaN-prefilled and cache positions past each slot's length are
//!   NaN, so a missing write or a read past the length fails. The appended positions must
//!   hold the new keys and values bit for bit, every other position of a real slot stay
//!   untouched and the device lengths stay as they were. Batches of one-token rows, and
//!   batches of uniform rows whose split plan matches, must give
//!   [`DecoderAttn`]'s outputs and caches bit for bit. Also `kv_lens_set` through
//!   [`Workspace::sync_lens`] and `gather_rows_f16x8`'s copied, zero and NaN rows
//! - `verify_matches_torch_and_the_step` (also `WHISPERDRZ_WEIGHTS` and
//!   `WHISPERDRZ_DECODER_REF`): the traced short3 prefill and two decode steps of
//!   `tests/gpu_decoder.rs`, each run as one verify step, every block, the final LayerNorm
//!   and the logits judged against the FP32 chain on the same bars as the chained step
//!   (mean-abs at 1.5x torch FP16's, the argmax at each row's last position equal to torch
//!   FP16's). The decode steps as verify steps on the step's own caches must give the
//!   step's residual, LayerNorm and logits bit for bit. Steps 1 and 2 fused into one
//!   verify step (two-token rows for the slots in both) are judged against both steps'
//!   truth, and must repeat bit for bit
//!
//! Run through the GPU queue with `rust/scripts/gpu-test.sh`
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
use numerics::{NAN16, Rng, f16_to_f64, gpu_enabled, ulp16};
use parity::{Check, Row, bits_f64, download16, f64s, report};
use serde_json::Value;
use whisperdrz_fixtures::{ArrayData, TraceSet};
use whisperdrz_native::cuda::decoder::verify::VerifyState;
use whisperdrz_native::cuda::decoder::{
    D_MODEL, MAX_ROWS, N_LAYER, N_VOCAB, PREFILL_LEN, Workspace,
};
use whisperdrz_native::cuda::decoder_attn::verify::{MAX_QUERIES, Ragged, VerifyAttn};
use whisperdrz_native::cuda::decoder_attn::{
    Batch, CROSS_LEN, HEAD_DIM, HEADS, Rows, SCALE, SELF_CAP, STATE,
};
use whisperdrz_native::cuda::{
    Arg, Blas, Buf, Decoder, DecoderAttn, Gpu, KvCaches, Weights, blocks,
};
use whisperdrz_native::transcriber::{VerifyRow, VerifyStep};

/// Absolute slack on top of one FP16 ulp against f64: FP32 rounding of the softmax sums,
/// which only matters for outputs near zero, as in `tests/gpu_decoder_attn.rs`
const F32_SLACK: f64 = 1.0 / (1 << 22) as f64;
/// A NaN the kernels never write, for prefilled outputs
const PREFILL_NAN: u16 = 0x7fff;

/// FP16 outputs against f64: the worst ulp error and the count past one ulp plus slack
#[derive(Default)]
struct UlpStats {
    max_ulp: f64,
    over: usize,
    n: usize,
}

impl UlpStats {
    fn add(&mut self, got: u16, want: f64) {
        let g = f16_to_f64(got);
        self.n += 1;
        if !g.is_finite() {
            self.max_ulp = f64::INFINITY;
            self.over += 1;
            return;
        }

        let err = (g - want).abs();
        self.max_ulp = self.max_ulp.max(err / ulp16(want));
        if err > ulp16(want) + F32_SLACK {
            self.over += 1;
        }
    }
}

/// One query and head of attention in f64: `key(j)` gives the K and V head rows of key
/// `j < n`
fn attend_f64<'a>(
    q: &[u16],
    n: usize,
    key: impl Fn(usize) -> (&'a [u16], &'a [u16]),
) -> [f64; HEAD_DIM] {
    let scores: Vec<f64> = (0..n)
        .map(|j| {
            let (k, _) = key(j);
            let s: f64 = q
                .iter()
                .zip(k)
                .map(|(&a, &b)| f16_to_f64(a) * f16_to_f64(b))
                .sum();
            s * SCALE as f64
        })
        .collect();
    let mx = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut out = [0f64; HEAD_DIM];
    let mut sum = 0.0;
    for (j, s) in scores.iter().enumerate() {
        let p = (s - mx).exp();
        sum += p;
        let (_, v) = key(j);
        for (o, &x) in out.iter_mut().zip(v) {
            *o += p * f16_to_f64(x);
        }
    }

    out.map(|o| o / sum)
}

/// Offset of head `hd` of position `j` of slot `s` in a `[slots, cap, STATE]` cache
fn at(s: usize, j: usize, cap: usize, hd: usize) -> usize {
    (s * cap + j) * STATE + hd * HEAD_DIM
}

/// One synthetic verify launch: real rows on distinct slots, then `pad` one-token padding
/// rows on the trash slot at length 0. Slot `n_slots` is the trash slot
struct Case {
    name: String,
    is_self: bool,
    q_lens: Vec<usize>,
    slots: Vec<usize>,
    pad: usize,
    n_slots: usize,
    /// Every slot's cache length, the trash slot's 0
    slot_lens: Vec<usize>,
    k_cache: Vec<u16>,
    v_cache: Vec<u16>,
    /// `[tokens, 3 * STATE]` packed Q, K and V, as the QKV linear writes them
    qkv: Vec<u16>,
}

impl Case {
    /// Random inputs: real slots in reverse order plus one slot no row uses, so the
    /// row-to-slot mapping isn't the identity and an untouched slot is checked; the first
    /// row's length is 0 and the second's fills the cache to the cap
    fn new(name: &str, is_self: bool, q_lens: &[usize], pad: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let real = q_lens.len();
        let n_slots = real + 1;
        let slots: Vec<usize> = (0..real).rev().collect();
        let cap = if is_self { SELF_CAP } else { CROSS_LEN };
        let mut slot_lens = vec![0; n_slots + 1];
        slot_lens[real] = 5;
        for (r, &q) in q_lens.iter().enumerate() {
            let most = SELF_CAP - q;
            slot_lens[slots[r]] = match r {
                0 => 0,
                1 => most,
                _ => (rng.uniform() * most as f64) as usize,
            };
        }

        let cache = (n_slots + 1) * cap * STATE;
        let mut k_cache = rng.f16_normal(cache, 0.0, 1.0);
        let mut v_cache = rng.f16_normal(cache, 0.0, 1.0);
        if is_self {
            for (s, &len) in slot_lens.iter().enumerate() {
                k_cache[at(s, len, cap, 0)..at(s + 1, 0, cap, 0)].fill(NAN16);
                v_cache[at(s, len, cap, 0)..at(s + 1, 0, cap, 0)].fill(NAN16);
            }
        }
        let tokens = q_lens.iter().sum::<usize>() + pad;
        let mut qkv = rng.f16_normal(tokens * 3 * STATE, 0.0, 1.0);
        // decode-like query scale, so the softmax is neither flat nor one-hot
        for t in 0..tokens {
            for v in &mut qkv[t * 3 * STATE..][..STATE] {
                *v = f16::from_f64(f16_to_f64(*v) * 0.5).to_bits();
            }
        }

        Self {
            name: name.to_owned(),
            is_self,
            q_lens: q_lens.to_vec(),
            slots,
            pad,
            n_slots,
            slot_lens,
            k_cache,
            v_cache,
            qkv,
        }
    }

    fn cap(&self) -> usize {
        if self.is_self { SELF_CAP } else { CROSS_LEN }
    }

    fn trash(&self) -> usize {
        self.n_slots
    }

    fn real_tokens(&self) -> usize {
        self.q_lens.iter().sum()
    }

    fn tokens(&self) -> usize {
        self.real_tokens() + self.pad
    }

    fn rows(&self) -> usize {
        self.q_lens.len() + self.pad
    }

    /// Row `r`'s first token, for every row and the closing one
    fn starts(&self) -> Vec<usize> {
        let mut s = vec![0];
        for q in self
            .q_lens
            .iter()
            .copied()
            .chain(std::iter::repeat_n(1, self.pad))
        {
            s.push(s.last().copied().unwrap_or(0) + q);
        }
        s
    }

    fn row_slots(&self) -> Vec<usize> {
        let trash = self.trash();
        self.slots
            .iter()
            .copied()
            .chain(std::iter::repeat_n(trash, self.pad))
            .collect()
    }

    /// Head `hd` of token `t`'s query, key or value (`part` 0, 1, 2)
    fn new_row(&self, part: usize, t: usize, hd: usize) -> &[u16] {
        &self.qkv[t * 3 * STATE + part * STATE + hd * HEAD_DIM..][..HEAD_DIM]
    }

    /// f64 attention of token `t` (row `r`'s query `i`), head `hd`
    fn reference(&self, r: usize, i: usize, t: usize, hd: usize) -> [f64; HEAD_DIM] {
        let s = self.row_slots()[r];
        let cap = self.cap();
        let len = if self.is_self { self.slot_lens[s] } else { cap };
        let n = if self.is_self { len + i + 1 } else { cap };
        let first = t - i;
        attend_f64(self.new_row(0, t, hd), n, |j| {
            if j < len {
                let a = at(s, j, cap, hd);
                (
                    &self.k_cache[a..][..HEAD_DIM],
                    &self.v_cache[a..][..HEAD_DIM],
                )
            } else {
                let tj = first + j - len;
                (self.new_row(1, tj, hd), self.new_row(2, tj, hd))
            }
        })
    }

    /// The self caches after the step: each real row's new keys and values at its slot's
    /// length plus the token's index in the row
    fn want_caches(&self) -> (Vec<u16>, Vec<u16>) {
        let (mut k, mut v) = (self.k_cache.clone(), self.v_cache.clone());
        let starts = self.starts();
        for (r, &s) in self.slots.iter().enumerate() {
            for i in 0..self.q_lens[r] {
                let t = starts[r] + i;
                let dst = at(s, self.slot_lens[s] + i, SELF_CAP, 0);
                k[dst..dst + STATE].copy_from_slice(&self.qkv[t * 3 * STATE + STATE..][..STATE]);
                v[dst..dst + STATE]
                    .copy_from_slice(&self.qkv[t * 3 * STATE + 2 * STATE..][..STATE]);
            }
        }
        (k, v)
    }
}

/// A launch's outputs: the attention output, the layer's caches and the device lengths
struct Outcome {
    out: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
    lens: Vec<i32>,
}

/// Fresh caches holding the case's, and the case's packed QKV on the device
fn upload_case(gpu: &Gpu, c: &Case) -> Result<(KvCaches, Buf)> {
    let caches = KvCaches::new(gpu, 1, c.n_slots)?;
    let (k_dst, v_dst) = if c.is_self {
        caches.self_slot(0, 0)?
    } else {
        caches.cross_slot(0, 0)?
    };
    gpu.upload_to(k_dst, bytemuck::cast_slice(&c.k_cache))?;
    gpu.upload_to(v_dst, bytemuck::cast_slice(&c.v_cache))?;
    let lens: Vec<i32> = c.slot_lens.iter().map(|&l| l as i32).collect();
    gpu.upload_to(caches.lens().ptr(), bytemuck::cast_slice(&lens))?;
    Ok((caches, gpu.upload(c.qkv.clone())?))
}

fn outcome(gpu: &Gpu, c: &Case, caches: &KvCaches, out: &Buf) -> Result<Outcome> {
    let (k_src, v_src) = if c.is_self {
        caches.self_slot(0, 0)?
    } else {
        caches.cross_slot(0, 0)?
    };
    let n = c.k_cache.len();
    Ok(Outcome {
        out: download16(gpu, out, c.tokens() * STATE)?,
        k: download_at(gpu, k_src, n)?,
        v: download_at(gpu, v_src, n)?,
        lens: gpu.download(caches.lens())?,
    })
}

/// `n` FP16 values of device memory at an address, through a scratch buffer
fn download_at(gpu: &Gpu, src: u64, n: usize) -> Result<Vec<u16>> {
    let tmp = gpu.alloc(n * 2)?;
    gpu.copy_dtod(src, tmp.ptr(), n * 2)?;
    download16(gpu, &tmp, n)
}

fn i32s(v: &[usize]) -> Vec<i32> {
    v.iter().map(|&x| x as i32).collect()
}

/// The case through [`VerifyAttn`], Q/K/V read from the packed buffer at its stride
fn run_verify(gpu: &Gpu, attn: &VerifyAttn, c: &Case) -> Result<Outcome> {
    let (caches, qkv) = upload_case(gpu, c)?;
    let row_slots = c.row_slots();
    let lens: Vec<usize> = row_slots
        .iter()
        .enumerate()
        .map(|(r, &s)| {
            if r < c.q_lens.len() {
                c.slot_lens[s]
            } else {
                0
            }
        })
        .collect();
    let (slots_d, starts_d, lens_d) = (
        gpu.upload(i32s(&row_slots))?,
        gpu.upload(i32s(&c.starts()))?,
        gpu.upload(i32s(&lens))?,
    );
    let rows = Ragged {
        slots: slots_d.ptr(),
        starts: starts_d.ptr(),
        lens: lens_d.ptr(),
        rows: c.rows(),
        tokens: c.tokens(),
        max_q: c.q_lens.iter().copied().max().unwrap_or(1),
    };
    let out = gpu.upload(vec![PREFILL_NAN; c.tokens() * STATE])?;
    let part = |p: usize| Rows {
        ptr: qkv.at(p * STATE * 2),
        stride: 3 * STATE,
    };
    if c.is_self {
        attn.self_attn(
            gpu,
            &caches,
            0,
            rows,
            (part(0), part(1), part(2)),
            out.ptr(),
        )?;
    } else {
        attn.cross_attn(gpu, &caches, 0, rows, part(0), out.ptr())?;
    }

    outcome(gpu, c, &caches, &out)
}

/// A case of uniform rows through [`DecoderAttn`] at that query count
fn run_step_attn(gpu: &Gpu, attn: &DecoderAttn, c: &Case, q_len: usize) -> Result<Outcome> {
    let (caches, qkv) = upload_case(gpu, c)?;
    let slots_d = gpu.upload(i32s(&c.row_slots()))?;
    let batch = Batch {
        slots: slots_d.ptr(),
        rows: c.rows(),
        q_len,
    };
    let out = gpu.upload(vec![PREFILL_NAN; c.tokens() * STATE])?;
    let part = |p: usize| Rows {
        ptr: qkv.at(p * STATE * 2),
        stride: 3 * STATE,
    };
    if c.is_self {
        attn.self_attn(
            gpu,
            &caches,
            0,
            batch,
            (part(0), part(1), part(2)),
            out.ptr(),
        )?;
    } else {
        attn.cross_attn(gpu, &caches, 0, batch, part(0), out.ptr())?;
    }

    outcome(gpu, c, &caches, &out)
}

/// Checks a verify launch against f64 and the cache contract; returns a failure line
/// per broken check
fn check_verify(c: &Case, got: &Outcome) -> Vec<String> {
    let mut failed = Vec::new();
    let starts = c.starts();
    let mut stats = UlpStats::default();
    for (r, &q) in c.q_lens.iter().enumerate() {
        for i in 0..q {
            let t = starts[r] + i;
            for hd in 0..HEADS {
                let want = c.reference(r, i, t, hd);
                let o = &got.out[t * STATE + hd * HEAD_DIM..][..HEAD_DIM];
                for (&g, &w) in o.iter().zip(&want) {
                    stats.add(g, w);
                }
            }
        }
    }
    let pads_finite = got.out[c.real_tokens() * STATE..]
        .iter()
        .all(|&v| f16_to_f64(v).is_finite());
    println!(
        "{:<26} {} rows ({} padding), {} tokens: max {:.3} ulp, {} of {} over 1 ulp",
        c.name,
        c.rows(),
        c.pad,
        c.tokens(),
        stats.max_ulp,
        stats.over,
        stats.n
    );
    if stats.over > 0 || stats.n != c.real_tokens() * STATE {
        failed.push(format!(
            "{}: {} outputs over 1 ulp of f64",
            c.name, stats.over
        ));
    }
    if !pads_finite {
        failed.push(format!("{}: a padding row's output isn't finite", c.name));
    }

    // the trash slot's contents are skipped: padding rows race on its position 0
    let real = at(c.trash(), 0, c.cap(), 0);
    let (want_k, want_v) = if c.is_self {
        c.want_caches()
    } else {
        (c.k_cache.clone(), c.v_cache.clone())
    };
    let diff = |a: &[u16], b: &[u16]| {
        a[..real]
            .iter()
            .zip(&b[..real])
            .filter(|(x, y)| x != y)
            .count()
    };
    let (dk, dv) = (diff(&got.k, &want_k), diff(&got.v, &want_v));
    if dk + dv > 0 {
        failed.push(format!(
            "{}: {dk} K and {dv} V cache elements differ from the expected append",
            c.name
        ));
    }
    let want_lens: Vec<i32> = c.slot_lens.iter().map(|&l| l as i32).collect();
    if got.lens != want_lens {
        failed.push(format!(
            "{}: device lengths {:?}, want them untouched {want_lens:?}",
            c.name, got.lens
        ));
    }

    failed
}

/// Checks `kv_lens_set` through [`Workspace::sync_lens`]: each staged row's slot gets the
/// row's position, the trash slot 0, other slots keep theirs
fn check_lens_sync(gpu: &Gpu) -> Result<Vec<String>> {
    let caches = KvCaches::new(gpu, 1, 5)?;
    let before = [11, 22, 33, 44, 55, 66];
    gpu.upload_to(caches.lens().ptr(), bytemuck::cast_slice(&before))?;
    let mut ws = Workspace::new(gpu, 8, 1, 0)?;
    let slots = [3, 0, 5, 2];
    let positions = [7, 120, 9, 0];
    let batch = ws.stage(gpu, &[0; 4], &positions, &slots)?;
    ws.sync_lens(gpu, &caches, batch)?;
    let got: Vec<i32> = gpu.download(caches.lens())?;
    let want = vec![120, 22, 0, 7, 55, 0];
    Ok(if got == want {
        println!("kv_lens_set: lengths {got:?}");
        Vec::new()
    } else {
        vec![format!("kv_lens_set: lengths {got:?}, want {want:?}")]
    })
}

/// Checks `gather_rows_f16x8`: copied rows bit for bit, -1 a zero row, past the source a
/// NaN row
fn check_gather(gpu: &Gpu) -> Result<Vec<String>> {
    let (n, c8) = (6, D_MODEL / 8);
    let src = Rng::new(7).f16_normal(n * D_MODEL, 0.0, 1.0);
    let idx: Vec<i32> = vec![2, -1, 5, 0, 6, -7, i32::MAX, 3];
    let rows = idx.len();
    let src_d = gpu.upload(src.clone())?;
    let idx_d = gpu.upload(idx.clone())?;
    let out = gpu.upload(vec![PREFILL_NAN; rows * D_MODEL])?;
    gpu.launch(
        "gather_rows_f16x8",
        (blocks(rows * c8, 256), 1, 1),
        (256, 1, 1),
        0,
        &[
            Arg::slice(out.ptr(), rows * c8),
            Arg::slice(src_d.ptr(), n * c8),
            Arg::slice(idx_d.ptr(), rows),
            c8.into(),
        ],
    )?;
    let got = download16(gpu, &out, rows * D_MODEL)?;

    let mut failed = Vec::new();
    for (i, &s) in idx.iter().enumerate() {
        let row = &got[i * D_MODEL..][..D_MODEL];
        let ok = match usize::try_from(s) {
            Err(_) => row.iter().all(|&v| v == 0),
            Ok(s) if s < n => row == &src[s * D_MODEL..][..D_MODEL],
            Ok(_) => row.iter().all(|&v| v == NAN16),
        };
        if !ok {
            failed.push(format!("gather_rows_f16x8: row {i} (index {s})"));
        }
    }
    Ok(failed)
}

#[test]
fn verify_kernels_match_f64() -> Result<()> {
    if !gpu_enabled() {
        return Ok(());
    }
    let gpu = Gpu::new(0)?;
    let attn = DecoderAttn::new(&gpu)?;
    let verify = VerifyAttn::new(&gpu, 64 * MAX_QUERIES)?;
    let mut failed = Vec::new();

    // ragged rows: every kernel variant, few rows (many splits) to many (one split)
    let ragged: [(&str, &[usize], usize); 5] = [
        ("q2", &[1, 2, 2], 1),
        ("q4", &[3, 1, 4, 2, 4], 2),
        ("q8", &[8, 1, 5, 7, 2, 6, 3], 3),
        ("q8 one per row", &[8], 0),
        (
            "q8 41 rows",
            &[
                1, 8, 3, 5, 2, 7, 4, 6, 1, 1, 2, 3, 8, 8, 5, 1, 2, 4, 6, 7, 3, 1, 1, 2, 5, 8, 3, 2,
                1, 4, 6, 1, 1, 7, 2, 3, 5, 1, 8, 2, 4,
            ],
            5,
        ),
    ];
    let mut seed = 100;
    for is_self in [true, false] {
        for (name, q_lens, pad) in ragged {
            seed += 1;
            let kind = if is_self { "self" } else { "cross" };
            let c = Case::new(&format!("{kind} {name}"), is_self, q_lens, pad, seed);
            let plan = if is_self {
                verify.self_plan(c.rows())
            } else {
                verify.cross_plan(c.rows())
            };
            print!("[{} splits] ", plan.splits);
            failed.extend(check_verify(&c, &run_verify(&gpu, &verify, &c)?));
        }
    }

    // uniform rows: the one-token step's attention bit for bit where the plans agree, which
    // for one-token rows is always and for wider rows once a single split covers the keys
    let mut bitwise = Vec::new();
    for is_self in [true, false] {
        let kind = if is_self { "self" } else { "cross" };
        for (q, rows, pad) in [
            (1, 1, 1),
            (1, 3, 1),
            (1, 9, 0),
            (1, 40, 2),
            (2, 0, 0),
            (3, 0, 0),
            (7, 0, 0),
        ] {
            let plans = |r: usize| {
                if is_self {
                    (verify.self_plan(r), attn.self_plan(r, q))
                } else {
                    (verify.cross_plan(r), attn.cross_plan(r, q))
                }
            };
            // wider rows: the fewest rows whose plans agree
            let rows = if rows > 0 {
                rows
            } else {
                (1..=64)
                    .find(|&r| {
                        let (a, b) = plans(r);
                        a == b
                    })
                    .ok_or_else(|| eyre!("no row count up to 64 where {q}-token plans agree"))?
            };
            seed += 1;
            let q_lens = vec![q; rows];
            let c = Case::new(
                &format!("{kind} uniform q{q} x{rows}"),
                is_self,
                &q_lens,
                pad,
                seed,
            );
            let (vp, sp) = plans(c.rows());
            ensure!(vp == sp, "{}: plans {vp:?} and {sp:?}", c.name);
            print!("[{} splits] ", vp.splits);
            let got = run_verify(&gpu, &verify, &c)?;
            failed.extend(check_verify(&c, &got));
            let step = run_step_attn(&gpu, &attn, &c, q)?;
            let real = c.real_tokens() * STATE;
            let trash = at(c.trash(), 0, c.cap(), 0);
            let same = got.out[..real] == step.out[..real]
                && got.k[..trash] == step.k[..trash]
                && got.v[..trash] == step.v[..trash];
            bitwise.push((c.name.clone(), same));
            if !same {
                failed.push(format!(
                    "{}: outputs or caches differ from the one-token step's attention",
                    c.name
                ));
            }
        }
    }
    println!(
        "bit for bit with DecoderAttn: {} of {}",
        bitwise.iter().filter(|(_, s)| *s).count(),
        bitwise.len()
    );

    failed.extend(check_lens_sync(&gpu)?);
    failed.extend(check_gather(&gpu)?);
    ensure!(failed.is_empty(), "verify kernels: {failed:#?}");
    Ok(())
}

struct Env {
    refs: PathBuf,
    manifest: Value,
    gpu: Gpu,
    blas: Blas,
    attn: DecoderAttn,
    decoder: Decoder,
    /// The traced prefill's encoder output, `[5 * 1500, 1280]`
    xa: Buf,
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
    Ok(Some(Env {
        refs,
        manifest,
        gpu,
        blas,
        attn,
        decoder,
        xa,
    }))
}

fn json_list<T>(v: &Value, f: impl Fn(&Value) -> Option<T>) -> Result<Vec<T>> {
    v.as_array()
        .ok_or_else(|| eyre!("{v} is not a list"))?
        .iter()
        .map(|x| f(x).ok_or_else(|| eyre!("bad list item {x}")))
        .collect()
}

/// One traced phase's inputs and its directory of torch outputs
struct Phase {
    name: &'static str,
    dir: PathBuf,
    tokens: Vec<u32>,
    positions: Vec<i32>,
    slots: Vec<usize>,
    q_len: usize,
}

impl Phase {
    fn load(env: &Env, name: &'static str) -> Result<Self> {
        let p = &env.manifest["phases"][name];
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
            name,
            dir: env.refs.join(name),
            q_len: tokens.len() / slots.len(),
            tokens,
            positions,
            slots,
        })
    }

    fn rows(&self) -> usize {
        self.slots.len()
    }

    /// The real rows, which lead; padding rows on the trash slot trail
    fn real_rows(&self, trash: usize) -> usize {
        self.slots.iter().take_while(|&&s| s != trash).count()
    }

    /// The whole phase as one verify step: each row its tokens at its positions, every
    /// token's logits read, no padding added
    fn verify_rows(&self) -> Vec<VerifyRow<'_>> {
        (0..self.rows())
            .map(|r| VerifyRow {
                slot: self.slots[r],
                kv_len: self.positions[r * self.q_len] as usize,
                fed: &self.tokens[r * self.q_len..][..self.q_len],
                forced: 0,
            })
            .collect()
    }

    /// A torch output of this phase, `kind` one of `f16`, `f32` and the FP16 variants
    fn torch(&self, kind: &str, name: &str) -> Result<Vec<f64>> {
        f64s(&self.dir.join(format!("{kind}_{name}.npy")))
    }
}

/// A verify step's per-layer outputs over its packed tokens
struct Layers {
    /// `[N_LAYER][tokens * D_MODEL]` residual after each block
    blocks: Vec<Vec<u16>>,
    ln: Vec<u16>,
    /// `[need, N_VOCAB]`
    logits: Vec<u16>,
}

impl Layers {
    /// The output of layer `name` (`blockNN`, `ln`, `logits`) and its row width
    fn get(&self, name: &str) -> Result<(&[u16], usize)> {
        Ok(match name {
            "ln" => (&self.ln, D_MODEL),
            "logits" => (&self.logits, N_VOCAB),
            b => {
                let i: usize = b.trim_start_matches("block").parse()?;
                (&self.blocks[i], D_MODEL)
            }
        })
    }
}

/// Runs `step` through [`Decoder::verify_traced`] with NaN-prefilled outputs and
/// downloads every layer
fn run_verify_step(
    env: &Env,
    vs: &VerifyState,
    caches: &KvCaches,
    ws: &mut Workspace,
    step: &VerifyStep,
) -> Result<Layers> {
    let gpu = &env.gpu;
    let n = step.bucket();
    let nan =
        |buf: &Buf, n: usize| gpu.upload_to(buf.ptr(), bytemuck::cast_slice(&vec![PREFILL_NAN; n]));
    nan(ws.x(), n * D_MODEL)?;
    nan(ws.h(), n * D_MODEL)?;
    nan(ws.logits(), step.head_rows * N_VOCAB)?;
    let staged = vs.stage(gpu, step, step.trash_slot)?;
    let mut blocks = Vec::with_capacity(N_LAYER);
    env.decoder
        .verify_traced(gpu, &env.blas, &vs.attn, caches, ws, &staged, |_, ws| {
            blocks.push(download16(gpu, ws.x(), n * D_MODEL)?);
            Ok(())
        })?;
    Ok(Layers {
        blocks,
        ln: download16(gpu, ws.h(), n * D_MODEL)?,
        logits: download16(gpu, ws.logits(), step.need.len() * N_VOCAB)?,
    })
}

/// Which traced row each verify token stands for: (phase, token index in that phase)
type TokenMap = Vec<(usize, usize)>;

/// Judges each layer of `ours` (tokens in `map` order) against the FP32 chain of the
/// phases `map` points into, on the chained step's bars: mean-abs at 1.5x torch FP16's,
/// max-abs against the worst torch FP16 variant, printed. Returns the failing rows and
/// the argmax mismatches with torch FP16 at the tokens in `argmax_at`
fn judge(
    title: &str,
    names: &[String],
    phases: &[&Phase],
    map: &TokenMap,
    ours: &Layers,
    argmax_at: &[usize],
) -> Result<Vec<String>> {
    let mut rows = Vec::new();
    let mut failed = Vec::new();
    for name in names.iter().filter(|n| *n != "embed") {
        let (bits, width) = ours.get(name)?;
        let mut loaded: Vec<[Vec<f64>; 5]> = Vec::new();
        for p in phases {
            loaded.push([
                p.torch("f16", name)?,
                p.torch("f32", name)?,
                p.torch("f16qkv", name)?,
                p.torch("f16sdpa", name)?,
                p.torch("f16eff", name)?,
            ]);
        }
        let pick = |k: usize| -> Vec<f64> {
            map.iter()
                .flat_map(|&(ph, t)| loaded[ph][k][t * width..][..width].iter().copied())
                .collect()
        };
        let ours = bits_f64(&bits[..map.len() * width]);
        let (t16, truth) = (pick(0), pick(1));
        let variants = [
            Row::new(name, &pick(2), &t16, &truth)?,
            Row::new(name, &pick(3), &t16, &truth)?,
            Row::new(name, &pick(4), &t16, &truth)?,
        ];
        rows.push(Row::new(name, &ours, &t16, &truth)?.bar_from(&[
            &variants[0],
            &variants[1],
            &variants[2],
        ]));

        if name == "logits" {
            for &i in argmax_at {
                let seg = |v: &[f64]| argmax(&v[i * N_VOCAB..][..N_VOCAB]);
                let (a, a16, a32) = (seg(&ours), seg(&t16), seg(&truth));
                println!("{title}: token {i} argmax ours {a}, torch FP16 {a16}, FP32 {a32}");
                if a != a16 {
                    failed.push(format!("{title} token {i}: argmax {a} vs torch {a16}"));
                }
            }
        }
    }

    failed.extend(report(
        &format!("{title} (truth: FP32 chain)"),
        &rows,
        Check::Mean,
    ));
    Ok(failed)
}

fn argmax(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0
}

#[test]
fn verify_matches_torch_and_the_step() -> Result<()> {
    let Some(env) = env()? else { return Ok(()) };
    let (gpu, blas, attn, decoder) = (&env.gpu, &env.blas, &env.attn, &env.decoder);
    let names: Vec<String> = json_list(&env.manifest["names"], |v| v.as_str().map(str::to_owned))?;
    ensure!(
        names.len() == N_LAYER + 3,
        "{} layer names, want embed + 32 blocks + ln + logits",
        names.len()
    );

    let slots_n = 5;
    let phases = [
        Phase::load(&env, "prefill")?,
        Phase::load(&env, "step1")?,
        Phase::load(&env, "step2")?,
    ];
    let vs = VerifyState::new(gpu, MAX_ROWS)?;
    // the step chain; the verify chain from its own verify prefill, judged against torch;
    // the step's prefill then verify steps, which must give the step's bits
    let step_kv = KvCaches::new(gpu, N_LAYER, slots_n)?;
    let ver_kv = KvCaches::new(gpu, N_LAYER, slots_n)?;
    let bit_kv = KvCaches::new(gpu, N_LAYER, slots_n)?;
    let mut ws_step = Workspace::new(gpu, MAX_ROWS, PREFILL_LEN, 0)?;
    let mut ws = Workspace::new(gpu, MAX_ROWS, PREFILL_LEN, 0)?;

    let mut failed = Vec::new();
    for p in &phases {
        let n = p.tokens.len();
        let real = p.real_rows(slots_n);
        let step = VerifyStep::new(&p.verify_rows(), &[], slots_n);
        ensure!(
            step.bucket() == n && step.need.len() == n && step.head_rows == n,
            "{}: verify step of {} tokens, {} needed, {} head rows",
            p.name,
            step.bucket(),
            step.need.len(),
            step.head_rows
        );

        // the step path
        if p.name == "prefill" {
            decoder.prefill(
                gpu,
                blas,
                attn,
                &step_kv,
                &mut ws_step,
                env.xa.ptr(),
                &p.tokens,
                &p.slots,
            )?;
        } else {
            let b = ws_step.stage(gpu, &p.tokens, &p.positions, &p.slots)?;
            decoder.step(gpu, blas, attn, &step_kv, &mut ws_step, b)?;
        }
        let step_out = (
            download16(gpu, ws_step.x(), n * D_MODEL)?,
            download16(gpu, ws_step.h(), n * D_MODEL)?,
            download16(gpu, ws_step.logits(), n * N_VOCAB)?,
        );

        // the verify chain against torch, real rows only
        if p.name == "prefill" {
            decoder.fill_cross(gpu, blas, &ver_kv, &ws, env.xa.ptr(), &p.slots)?;
        }
        let ours = run_verify_step(&env, &vs, &ver_kv, &mut ws, &step)?;
        let map: TokenMap = (0..real * p.q_len).map(|t| (0, t)).collect();
        let last: Vec<usize> = (0..real).map(|r| r * p.q_len + p.q_len - 1).collect();
        failed.extend(judge(
            &format!("{} as a verify step", p.name),
            &names,
            &[p],
            &map,
            &ours,
            &last,
        )?);

        // the step's caches: the prefill as the step runs it, then verify steps
        if p.name == "prefill" {
            decoder.prefill(
                gpu,
                blas,
                attn,
                &bit_kv,
                &mut ws,
                env.xa.ptr(),
                &p.tokens,
                &p.slots,
            )?;
        } else {
            let got = run_verify_step(&env, &vs, &bit_kv, &mut ws, &step)?;
            let same = got.blocks[N_LAYER - 1] == step_out.0
                && got.ln == step_out.1
                && got.logits == step_out.2;
            println!(
                "{}: one-token verify step {} the step bit for bit",
                p.name,
                if same { "gives" } else { "DOESN'T give" }
            );
            if !same {
                failed.push(format!(
                    "{}: one-token verify step differs from the step",
                    p.name
                ));
            }
        }
    }

    // steps 1 and 2 fused: slots 0 and 4 feed both tokens as one two-token row, the rest
    // their one token, on the step's caches after the prefill (the verify steps above
    // only rewrote positions 7 and 8 with the step's own bits)
    let (s1, s2) = (&phases[1], &phases[2]);
    let row_of = |p: &Phase, slot: usize| p.slots.iter().position(|&s| s == slot);
    let mut fed: Vec<(usize, Vec<u32>, TokenMap)> = Vec::new();
    for slot in 0..slots_n {
        let mut tokens = Vec::new();
        let mut map = Vec::new();
        for (ph, p) in [(0, s1), (1, s2)] {
            if let Some(r) = row_of(p, slot) {
                ensure!(
                    p.positions[r] as usize == 7 + tokens.len(),
                    "slot {slot}: {} position {} doesn't follow its fused tokens",
                    p.name,
                    p.positions[r]
                );
                tokens.push(p.tokens[r]);
                map.push((ph, r));
            }
        }
        if !tokens.is_empty() {
            fed.push((slot, tokens, map));
        }
    }
    ensure!(
        fed.iter().any(|(_, t, _)| t.len() == 2),
        "no slot runs in both decode steps"
    );
    let rows: Vec<VerifyRow> = fed
        .iter()
        .map(|(slot, tokens, _)| VerifyRow {
            slot: *slot,
            kv_len: 7,
            fed: tokens,
            forced: 0,
        })
        .collect();
    let step = VerifyStep::new(&rows, &[], slots_n);
    let map: TokenMap = fed.iter().flat_map(|(_, _, m)| m.iter().copied()).collect();
    println!(
        "fused steps 1 and 2: rows {:?}",
        fed.iter()
            .map(|(s, t, _)| (*s, t.len()))
            .collect::<Vec<_>>()
    );
    let fused = run_verify_step(&env, &vs, &bit_kv, &mut ws, &step)?;
    let all: Vec<usize> = (0..map.len()).collect();
    failed.extend(judge(
        "steps 1 and 2 fused",
        &names,
        &[s1, s2],
        &map,
        &fused,
        &all,
    )?);

    let again = run_verify_step(&env, &vs, &bit_kv, &mut ws, &step)?;
    let repeats =
        again.blocks == fused.blocks && again.ln == fused.ln && again.logits == fused.logits;
    println!("fused verify step repeats bit for bit: {repeats}");
    if !repeats {
        failed.push("the fused verify step doesn't repeat bit for bit".to_owned());
    }

    ensure!(failed.is_empty(), "verify forward: {failed:#?}");
    Ok(())
}
