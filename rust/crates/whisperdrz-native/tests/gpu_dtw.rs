//! The native DTW forward and prep against the reference traces and torch
//!
//! `dtw_matches_torch` reruns every recorded short3 `dtw_decoder_forward` and
//! `_dtw_prep_batch` on the GPU: our raw alignment-head scores and prepped matrices
//! against the recorded torch FP16 ones, both judged against torch's FP32 forward and prep
//! on the same FP16 weights and inputs (`rust/tools/decoder_layers.py`), with the same
//! 1.5x rule as the layer checks. The forward runs 26 decoder layers before the capture,
//! so as for the chained layers, max-abs is judged against the worst of three torch FP16
//! runs exact in real arithmetic (the transcriber's; one packed QKV GEMM with a zero key
//! bias, the native layout; memory-efficient SDPA), and mean-abs against the transcriber's
//! alone; both bars are printed. It also checks the capture GEMM bit for bit on torch's
//! own query and key projections, and prints our prep alone on torch's scores.
//! `dtw_word_timings_match_reference` runs every `fill_word_timings_batch` call through
//! the CPU DTW with our matrices and with the recorded ones, and counts the words whose
//! start or end moves and by how many 0.02 s frames. Both read the recorded encoder
//! outputs through cross keys and values a prefill cached, as the transcriber does. They
//! skip unless `WHISPERDRZ_WEIGHTS` and `WHISPERDRZ_DECODER_REF` are set
//!
//! The forward reads the cross keys and values each slot's prefill cached, where the
//! reference projects them again over the DTW rows. `cross_kv_bits_ignore_row_count` checks
//! that cuBLAS gives a window the same bits at every row count either GEMM runs at, and
//! `dtw_cached_kv_matches_projection` runs the whole forward and prep on cached keys and
//! values against the reference's projection, bit for bit, over slot patterns that
//! exercise every launch grouping (`WHISPERDRZ_GPU=1`, the second also
//! `WHISPERDRZ_WEIGHTS`). `dtw_side_stream_is_fenced` runs the forward on a second stream
//! between a prefill and the next one into the same slots, queued with no host wait, and
//! wants the same bits as the forward on the caches' own stream. Run through the GPU queue
//! with `rust/scripts/gpu-test.sh`
#![cfg(feature = "cuda")]

#[allow(dead_code)]
#[path = "support/numerics.rs"]
mod numerics;
#[allow(dead_code)]
#[path = "support/parity.rs"]
mod parity;

use std::path::PathBuf;

use eyre::{Result, bail, ensure, eyre};
use numerics::{Err, NAN16, Rng};
use parity::{Check, MAX_RATIO, Row, f16_bits, f64s, report};
use serde_json::Value;
use whisperdrz_fixtures::{ArrayData, Event, TraceSet};
use whisperdrz_native::cuda::decoder::{
    ALIGNMENT_HEADS, D_MODEL, DtwCapture, DtwScores, N_AUDIO_CTX, N_LAYER, Workspace,
};
use whisperdrz_native::cuda::decoder_attn::XA_BYTES;
use whisperdrz_native::cuda::{Blas, Buf, Decoder, Gpu, KvCaches, Priority, Weights};
use whisperdrz_native::dtw::{DtwBatch, DtwOptions, Matrices, PreparedBatch, TokenizerDtw};
use whisperdrz_native::text::Tokenizer;

/// One DTW frame: 50 audio positions per second
const FRAME: f64 = 0.02;

struct Env {
    refs: PathBuf,
    manifest: Value,
    set: TraceSet,
    gpu: Gpu,
    blas: Blas,
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
    let Some(set) = TraceSet::open("short3") else {
        return Ok(None);
    };
    let manifest: Value = serde_json::from_slice(&std::fs::read(refs.join("manifest.json"))?)?;
    let gpu = Gpu::new(0)?;
    let blas = Blas::new(&gpu)?;
    let mut w = Weights::load(&gpu, &weights, |n| n.starts_with("decoder."))?;
    let decoder = Decoder::new(&gpu, &mut w)?;
    Ok(Some(Env {
        refs,
        manifest,
        set,
        gpu,
        blas,
        decoder,
    }))
}

fn usizes(v: &Value) -> Result<Vec<usize>> {
    v.as_array()
        .ok_or_else(|| eyre!("{v} is not a list"))?
        .iter()
        .map(|x| {
            x.as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| eyre!("bad item {x}"))
        })
        .collect()
}

/// An FP16 array on the device
fn upload16(gpu: &Gpu, set: &TraceSet, v: &Value) -> Result<(Vec<usize>, Buf)> {
    let a = set.array(v)?;
    let ArrayData::F16(data) = a.data else {
        bail!("array is not float16")
    };
    Ok((
        a.shape,
        gpu.upload(data.iter().map(|h| h.to_bits()).collect::<Vec<u16>>())?,
    ))
}

/// `[rows, heads, t, a]` (torch's layout) from our `[heads, rows, t, a]`
fn heads_inner(v: &[f32], (heads, rows, t): (usize, usize, usize)) -> Vec<f64> {
    let plane = t * N_AUDIO_CTX;
    let mut out = Vec::with_capacity(v.len());
    for b in 0..rows {
        for s in 0..heads {
            out.extend(
                v[(s * rows + b) * plane..][..plane]
                    .iter()
                    .map(|&x| f64::from(x)),
            );
        }
    }
    out
}

/// `[heads, rows, t, a]` from torch's `[rows, heads, t, a]`
fn heads_outer(v: &[f32], (heads, rows, t): (usize, usize, usize)) -> Vec<f32> {
    let plane = t * N_AUDIO_CTX;
    let mut out = Vec::with_capacity(v.len());
    for s in 0..heads {
        for b in 0..rows {
            out.extend_from_slice(&v[(b * heads + s) * plane..][..plane]);
        }
    }
    out
}

fn find(set: &TraceSet, stem: &str, label: &str, seq: u64) -> Result<Event> {
    set.calls(stem, label)?
        .into_iter()
        .find(|e| e.seq == seq)
        .ok_or_else(|| eyre!("{stem}: no {label} #{seq}"))
}

/// Raises the row's max-abs bar to the worst torch FP16 variant's error on `what`
/// (`scores` or `matrices`) recorded in the case
fn variant_bar(mut row: Row, case: &Value, what: &str) -> Result<Row> {
    let variants = case["variants"]
        .as_object()
        .ok_or_else(|| eyre!("{}: no torch FP16 variants in the manifest", row.name))?;
    ensure!(
        variants.len() == 2,
        "{}: {} variants",
        row.name,
        variants.len()
    );
    for (name, v) in variants {
        let max = v[what]["max_abs"]
            .as_f64()
            .ok_or_else(|| eyre!("{}: variant {name} has no {what} max-abs", row.name))?;
        row.max_bar = row.max_bar.max(max);
    }
    Ok(row)
}

fn f64v(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// The values at real tokens of a `[rows, planes, t, width]` array: token `i` of row `b`
/// counts below `lens[b]`. The DTW forward packs its shortest row's tokens, so that row's
/// padding isn't the reference's (and no alignment reads it)
fn real_tokens<T: Copy>(
    v: &[T],
    (rows, planes, t): (usize, usize, usize),
    lens: &[usize],
) -> Vec<T> {
    let width = v.len() / (rows * planes * t);
    assert_eq!(
        v.len(),
        rows * planes * t * width,
        "array of {} values",
        v.len()
    );
    assert_eq!(lens.len(), rows, "{} lengths for {rows} rows", lens.len());
    let mut out = Vec::new();
    for (i, plane) in v.chunks_exact(t * width).enumerate() {
        out.extend_from_slice(&plane[..lens[i / planes] * width]);
    }
    out
}

/// [`Decoder::fill_cross`] of the windows stacked at `xa` (FP16 `[slots.len() * 1500,
/// 1280]`) into `slots`, as a prefill of those windows fills them: the cross keys and
/// values the DTW forward reads
fn prefill(
    (gpu, blas, decoder): (&Gpu, &Blas, &Decoder),
    caches: &KvCaches,
    xa: u64,
    slots: &[usize],
) -> Result<()> {
    let ws = Workspace::new(gpu, 1, 1, slots.len())?;
    decoder.fill_cross(gpu, blas, caches, &ws, xa, slots)?;
    gpu.synchronize()
}

#[test]
fn dtw_matches_torch() -> Result<()> {
    let Some(env) = env()? else { return Ok(()) };
    let Env {
        refs,
        manifest,
        set,
        gpu,
        blas,
        decoder,
    } = &env;
    let cases = manifest["dtw"].as_array().cloned().unwrap_or_default();
    ensure!(
        cases.len() == 15,
        "{} DTW cases in the manifest, want 15",
        cases.len()
    );
    let heads = ALIGNMENT_HEADS.len();

    let (mut score_rows, mut matrix_rows) = (Vec::new(), Vec::new());
    let mut capture_checked = false;
    for case in &cases {
        let stem = case["stem"].as_str().unwrap_or_default();
        let fwd = find(
            set,
            stem,
            "dtw_decoder_forward",
            case["forward_seq"].as_u64().unwrap_or_default(),
        )?;
        let prep = find(
            set,
            stem,
            "_dtw_prep_batch",
            case["prep_seq"].as_u64().unwrap_or_default(),
        )?;
        let name = format!("{stem}#{}", fwd.seq);

        let x = set.array(fwd.arg("x")?)?;
        let [rows, t] = x.shape[..] else {
            bail!("{name}: x shape {:?}", x.shape)
        };
        let (xa_shape, xa) = upload16(gpu, set, fwd.arg("xa")?)?;
        ensure!(
            xa_shape == [rows, N_AUDIO_CTX, 1280],
            "{name}: xa shape {xa_shape:?}"
        );
        // one GEMM over the stacked rows, the row count the reference's projection runs at
        let caches = KvCaches::new(gpu, N_LAYER, rows)?;
        let slots: Vec<usize> = (0..rows).collect();
        prefill((gpu, blas, decoder), &caches, xa.ptr(), &slots)?;
        let batch = DtwBatch {
            full_tokens: x.to_i64()?.into_iter().map(|v| v as u32).collect(),
            t_pad: t,
            actual_lens: usizes(prep.arg("actual_lens")?)?,
            audio_caps: usizes(prep.arg("audio_caps")?)?,
            slots: (0..rows).collect(),
        };
        let torch_scores = set.array(prep.arg("full_weights")?)?;
        ensure!(
            torch_scores.shape == [rows, heads, t, N_AUDIO_CTX],
            "{name}: scores shape {:?}",
            torch_scores.shape
        );
        let torch_scores = torch_scores.to_f32()?;
        let recorded = set.array(prep.ret()?)?;
        let recorded_m = recorded.to_f32()?;

        // raw scores before the prep
        let scores = decoder.dtw_scores(gpu, blas, &caches, &batch, &slots)?;
        ensure!(
            scores.shape() == [heads, rows, t, N_AUDIO_CTX],
            "{name}: our scores {:?}",
            scores.shape()
        );
        let lens = &batch.actual_lens;
        let real = |v: &[f64]| real_tokens(v, (rows, heads, t), lens);
        let ours = real(&heads_inner(&scores.download(gpu)?, (heads, rows, t)));
        let torch16 = real(&f64v(&torch_scores));
        let same = ours.iter().zip(&torch16).filter(|(a, b)| a == b).count();
        let truth = real(&f64s(
            &refs.join(case["scores32"].as_str().unwrap_or_default()),
        )?);
        score_rows.push(variant_bar(
            Row::new(&name, &ours, &torch16, &truth)?,
            case,
            "scores",
        )?);
        println!(
            "{name}: B={rows} T={t} lens {:?} caps {:?}; scores equal to torch FP16 in {same}/{}; torch rerun vs recorded differ {}, flash-only vs default differ {}",
            batch.actual_lens,
            batch.audio_caps,
            ours.len(),
            case["rerun_differs"],
            case["flash_differs"]
        );

        // the prepped matrices
        let m = scores.prep(gpu, &batch)?;
        ensure!(
            m.shape[..] == recorded.shape[..],
            "{name}: matrices {:?} vs {:?}",
            m.shape,
            recorded.shape
        );
        let real_m = |v: &[f64]| real_tokens(v, (rows, 1, t), lens);
        let ours_m = real_m(&f64v(&m.data));
        let recorded_real = real_m(&f64v(&recorded_m));
        let direct = Err::of(&ours_m, &recorded_real)?;
        let truth_m = real_m(&f64s(
            &refs.join(case["matrices32"].as_str().unwrap_or_default()),
        )?);
        matrix_rows.push(variant_bar(
            Row::new(&name, &ours_m, &recorded_real, &truth_m)?,
            case,
            "matrices",
        )?);

        // the prep alone, on torch's own scores
        let only = DtwScores::upload(
            gpu,
            heads_outer(&torch_scores, (heads, rows, t)),
            (heads, rows, t),
        )?
        .prep(gpu, &batch)?;
        let prep_only = Err::of(&f64v(&only.data), &f64v(&recorded_m))?;
        println!(
            "  matrices vs recorded: max {:.4e} mean {:.4e}; our prep on torch's scores vs recorded: max {:.4e} mean {:.4e}",
            direct.max_abs, direct.mean_abs, prep_only.max_abs, prep_only.mean_abs
        );

        // the capture GEMM on torch's own FP16 projections
        if let Some(cap) = case.get("capture") {
            let mut capture = DtwCapture::new(gpu, rows, t)?;
            let mut keep = Vec::new();
            for (s, &(layer, head)) in ALIGNMENT_HEADS.iter().enumerate() {
                let (qs, q) =
                    f16_bits(&refs.join(cap["q"][layer.to_string()].as_str().unwrap_or_default()))?;
                let (ks, k) =
                    f16_bits(&refs.join(cap["k"][layer.to_string()].as_str().unwrap_or_default()))?;
                ensure!(
                    qs == [rows, t, 1280] && ks == [rows, N_AUDIO_CTX, 1280],
                    "capture shapes {qs:?} {ks:?}"
                );
                let (qd, kd) = (gpu.upload(q)?, gpu.upload(k)?);
                let keys: Vec<u64> = (0..rows).map(|b| kd.at(b * XA_BYTES)).collect();
                capture.gather(gpu, s, qd.ptr(), &keys, head)?;
                gpu.synchronize()?;
                keep.push((qd, kd));
            }
            let got = heads_inner(&capture.scores(gpu, blas)?.download(gpu)?, (heads, rows, t));
            let want = f64s(&refs.join(cap["scores16"].as_str().unwrap_or_default()))?;
            ensure!(
                got.len() == want.len(),
                "capture: {} scores, torch {}",
                got.len(),
                want.len()
            );
            let differ = got
                .iter()
                .zip(&want)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            println!(
                "  capture GEMM on torch's projections: {differ} of {} scores differ from torch's",
                got.len()
            );
            ensure!(
                differ == 0,
                "{name}: capture GEMM differs from torch in {differ} scores"
            );
            capture_checked = true;
        }
    }
    ensure!(capture_checked, "no capture case in the manifest");

    let mut failed = report(
        "DTW raw scores (truth: FP32 forward)",
        &score_rows,
        Check::Full,
    );
    failed.extend(report(
        "DTW prepped matrices (truth: FP32 forward and prep)",
        &matrix_rows,
        Check::Full,
    ));
    ensure!(
        failed.is_empty(),
        "worse than {MAX_RATIO}x torch FP16: {failed:?}"
    );
    Ok(())
}

/// Each `fill_word_timings_batch` call with the last `_dtw_prep_batch` recorded inside it
fn fill_calls(events: &[Event]) -> Vec<(&Event, Option<&Event>)> {
    // a call's nested events are emitted before it (seq is taken when a call returns)
    let mut out = Vec::new();
    let mut prep = None;
    for ev in events {
        match ev.ev.as_str() {
            "_dtw_prep_batch" => prep = Some(ev),
            "fill_word_timings_batch" => out.push((ev, prep.take())),
            _ => {}
        }
    }
    out
}

#[test]
fn dtw_word_timings_match_reference() -> Result<()> {
    let Some(env) = env()? else { return Ok(()) };
    let Env {
        set,
        gpu,
        blas,
        decoder,
        ..
    } = &env;
    let tokenizer = Tokenizer::multilingual()?;
    let tok = TokenizerDtw::new(&tokenizer)?;

    println!(
        "\n{:<14} {:>6} {:>6} {:>6} {:>6} {:>10} {:>6}",
        "call", "chunks", "words", "moved", "max fr", "frames", "texts"
    );
    let (mut words, mut moved, mut calls) = (0usize, 0usize, 0usize);
    let mut hist = std::collections::BTreeMap::<i64, usize>::new();
    let mut texts_differ = 0;
    for stem in set.stems()? {
        let events = set.events(&stem)?;
        for (fill, prep) in fill_calls(&events) {
            ensure!(
                fill.raise.is_none(),
                "{stem} #{}: the reference raised",
                fill.seq
            );
            let slot_data = fill
                .arg("slot_data")?
                .as_array()
                .ok_or_else(|| eyre!("slot_data is not a list"))?;
            let mut slots = Vec::new();
            let mut xa = Vec::new();
            for s in slot_data {
                let t = &s["__tuple"];
                let (shape, buf) = upload16(gpu, set, &t[0])?;
                ensure!(shape == [N_AUDIO_CTX, 1280], "slot xa shape {shape:?}");
                xa.push(buf);
                slots.push((
                    t[1].as_str().ok_or_else(|| eyre!("slot text"))?.to_owned(),
                    t[2].as_i64().ok_or_else(|| eyre!("num_frames"))?,
                ));
            }
            let prepared = PreparedBatch::new(&slots, &tok, &DtwOptions::default())?;
            let Some(batch) = prepared.batch() else {
                ensure!(
                    prep.is_none(),
                    "{stem} #{}: the reference ran a forward we don't",
                    fill.seq
                );
                continue;
            };
            let prep = prep.ok_or_else(|| eyre!("{stem} #{}: no recorded prep", fill.seq))?;
            // slot i holds slot_data[i]'s prefill, one window per GEMM
            let caches = KvCaches::new(gpu, N_LAYER, xa.len())?;
            for (i, w) in xa.iter().enumerate() {
                prefill((gpu, blas, decoder), &caches, w.ptr(), &[i])?;
            }
            let ours = decoder.dtw_prep(gpu, blas, &caches, batch, &batch.slots)?;
            prepared.check_matrices(ours.view())?;
            queued_matches(gpu, blas, decoder, &caches, batch, &ours)?;
            let rec = set.array(prep.ret()?)?;
            let recorded = Matrices {
                shape: [rec.shape[0], rec.shape[1], rec.shape[2]],
                data: rec.to_f32()?,
            };

            let (mut c_words, mut c_moved, mut c_max, mut c_frames) = (0, 0, 0i64, 0i64);
            let mut results = Vec::new();
            for bi in 0..batch.rows() {
                let (neg_o, neg_r) = (
                    prepared.chunk_neg_matrix(bi, ours.view())?,
                    prepared.chunk_neg_matrix(bi, recorded.view())?,
                );
                if neg_o.is_empty() {
                    results.push(Vec::new());
                    continue;
                }
                let (wo, wr) = (
                    prepared.chunk_dtw(bi, &neg_o),
                    prepared.chunk_dtw(bi, &neg_r),
                );
                ensure!(
                    wo.len() == wr.len(),
                    "chunk {bi}: {} words vs {}",
                    wo.len(),
                    wr.len()
                );
                for (a, b) in wo.iter().zip(&wr) {
                    let fr = |x: f64, y: f64| ((x - y) / FRAME).round().abs() as i64;
                    let (ds, de) = (fr(a.start, b.start), fr(a.end, b.end));
                    c_words += 1;
                    if ds > 0 || de > 0 {
                        c_moved += 1;
                        c_frames += ds + de;
                        c_max = c_max.max(ds.max(de));
                        *hist.entry(ds.max(de)).or_default() += 1;
                    }
                }
                results.push(wo);
            }
            let got = prepared.finish(&results)?;
            let want: Vec<String> = fill
                .ret()?
                .as_array()
                .ok_or_else(|| eyre!("fill returned no list"))?
                .iter()
                .map(|v| v.as_str().unwrap_or_default().to_owned())
                .collect();
            let c_texts = got.iter().zip(&want).filter(|(g, w)| g != w).count();
            println!(
                "{:<14} {:>6} {:>6} {:>6} {:>6} {:>10} {:>6}",
                format!("{stem}#{}", fill.seq),
                batch.rows(),
                c_words,
                c_moved,
                c_max,
                c_frames,
                format!("{c_texts}/{}", want.len())
            );
            words += c_words;
            moved += c_moved;
            texts_differ += c_texts;
            calls += 1;
        }
    }
    println!(
        "{calls} calls, {words} words: {moved} with a moved start or end; largest move per word (frames: words) {hist:?}; {texts_differ} slot texts differ from the recorded output"
    );
    ensure!(calls > 0 && words > 0, "no DTW calls in the trace");
    Ok(())
}

/// The transcriber's queued path gives `want` bit for bit: the prep left on the GPU,
/// copied into page-locked memory behind a sleeping event, waited on from another thread
fn queued_matches(
    gpu: &Gpu,
    blas: &Blas,
    decoder: &Decoder,
    caches: &KvCaches,
    batch: &DtwBatch,
    want: &Matrices,
) -> Result<()> {
    let (out, shape) = decoder.dtw_prep_queued(gpu, blas, caches, batch, &batch.slots)?;
    ensure!(
        shape == want.shape,
        "queued shape {shape:?}, want {:?}",
        want.shape
    );
    let host = gpu.alloc_host(out.bytes())?;
    let done = gpu.sleeping_event()?;
    // NaN first, so a copy that never lands can't pass
    let nan = vec![f32::NAN; out.bytes() / 4];
    let nan_buf = gpu.upload(nan)?;
    gpu.download_async(&nan_buf, 0, &host, 0, out.bytes(), &done)?;
    gpu.synchronize()?;
    gpu.download_async(&out, 0, &host, 0, out.bytes(), &done)?;
    drop(out);
    std::thread::spawn(move || done.synchronize_any_thread())
        .join()
        .map_err(|_| eyre!("waiting thread panicked"))??;
    // SAFETY: the copy finished at `done` and nothing else writes the buffer
    let got = unsafe { host.slice::<f32>(0, want.data.len())? };
    ensure!(
        got.iter()
            .map(|v| v.to_bits())
            .eq(want.data.iter().map(|v| v.to_bits())),
        "queued DTW matrices differ from dtw_prep's"
    );
    Ok(())
}

/// The GPU and cuBLAS handle, when `WHISPERDRZ_GPU=1`
fn gpu_only() -> Result<Option<(Gpu, Blas)>> {
    if std::env::var("WHISPERDRZ_GPU").as_deref() != Ok("1") {
        eprintln!("skipping: WHISPERDRZ_GPU=1 not set (run scripts/gpu-test.sh)");
        return Ok(None);
    }
    let gpu = Gpu::new(0)?;
    let blas = Blas::new(&gpu)?;
    Ok(Some((gpu, blas)))
}

/// FP16 values of one window's cross keys or values
const WINDOW: usize = N_AUDIO_CTX * D_MODEL;

/// Most windows one GEMM of the cross keys and values is checked at: a prefill takes up to
/// the 52 decode slots, and a DTW call rarely more than 9 rows
const MAX_WINDOWS: usize = 64;

/// The DTW forward reads the cross keys and values a prefill computed, over that
/// prefill's windows, where the reference projects them again over the DTW rows: GEMMs of
/// the same windows at other row counts and positions. This checks that cuBLAS gives each
/// window the same bits at every count from 1 to [`MAX_WINDOWS`] windows, and at the first,
/// middle and last position, for the bias-free key call and the value call with its bias
#[test]
fn cross_kv_bits_ignore_row_count() -> Result<()> {
    let Some((gpu, blas)) = gpu_only()? else {
        return Ok(());
    };
    let mut rng = Rng::new(0x5eed_d7a0);
    let xa = gpu.upload(rng.f16_normal(MAX_WINDOWS * WINDOW, 0.0, 1.0))?;
    let wk = gpu.upload(rng.f16_normal(D_MODEL * D_MODEL, 0.0, 0.03))?;
    let wv = gpu.upload(rng.f16_normal(D_MODEL * D_MODEL, 0.0, 0.03))?;
    let bv = gpu.upload(rng.f16_normal(D_MODEL, 0.0, 0.1))?;
    let out = gpu.alloc(MAX_WINDOWS * WINDOW * 2)?;
    out_nan(&gpu, &out)?;

    let project = |windows: usize, value: bool| -> Result<()> {
        let dims = (windows * N_AUDIO_CTX, D_MODEL, D_MODEL);
        if value {
            blas.linear_bias(xa.ptr(), wv.ptr(), bv.ptr(), out.ptr(), dims)
        } else {
            blas.linear(xa.ptr(), wk.ptr(), out.ptr(), dims)
        }
    };
    let window = |w: usize| -> Result<Vec<u16>> {
        let mut host = vec![0u16; WINDOW];
        gpu.download_into(&out, w * WINDOW * 2, &mut host)?;
        Ok(host)
    };

    let mut checked = 0;
    for value in [false, true] {
        // every window projected over all 64 at once is the reference; each count's
        // windows are compared with it
        project(MAX_WINDOWS, value)?;
        let all: Vec<Vec<u16>> = (0..MAX_WINDOWS).map(window).collect::<Result<_>>()?;
        ensure!(
            all.iter().flatten().all(|&b| b & 0x7c00 != 0x7c00),
            "non-finite cross {} at {MAX_WINDOWS} windows",
            if value { "value" } else { "key" }
        );
        for windows in 1..MAX_WINDOWS {
            out_nan(&gpu, &out)?;
            project(windows, value)?;
            let mut at = vec![0, windows / 2, windows - 1];
            at.dedup();
            for w in at {
                let got = window(w)?;
                ensure!(got.len() == all[w].len(), "window {w}: length");
                let differ = got.iter().zip(&all[w]).filter(|(a, b)| a != b).count();
                ensure!(
                    differ == 0,
                    "cross {} of window {w} at {windows} windows: {differ} of {WINDOW} values \
                     differ from the {MAX_WINDOWS}-window GEMM",
                    if value { "value" } else { "key" }
                );
                checked += 1;
            }
        }
    }
    println!("cross K/V: {checked} windows bit for bit at 1..{MAX_WINDOWS} windows");
    Ok(())
}

/// Fills a buffer with FP16 NaN, so a GEMM or kernel that skips a value fails the check
fn out_nan(gpu: &Gpu, buf: &Buf) -> Result<()> {
    gpu.upload_to(
        buf.ptr(),
        bytemuck::cast_slice(&vec![NAN16; buf.bytes() / 2]),
    )
}

/// NaN in slot `slot`'s encoder output and every layer's cross keys and values, so a DTW
/// row that reads it (a wrong stride or slot) turns its scores to NaN
fn poison_slot(gpu: &Gpu, caches: &KvCaches, slot: usize) -> Result<()> {
    let nan = vec![NAN16; WINDOW];
    let bytes: &[u8] = bytemuck::cast_slice(&nan);
    gpu.upload_to(caches.xa_slot(slot)?, bytes)?;
    for layer in 0..caches.layers() {
        let (k, v) = caches.cross_slot(layer, slot)?;
        gpu.upload_to(k, bytes)?;
        gpu.upload_to(v, bytes)?;
    }
    Ok(())
}

/// A DTW batch over `slots` of random text tokens, eot padding, lengths and audio caps
fn random_batch(rng: &mut Rng, slots: &[usize], t_pad: usize) -> DtwBatch {
    const EOT: u32 = 50257;
    let rows = slots.len();
    let mut full_tokens = vec![EOT; rows * t_pad];
    let mut actual_lens = Vec::with_capacity(rows);
    let mut audio_caps = Vec::with_capacity(rows);
    for b in 0..rows {
        // the first row fills t_pad, as the longest row of a real batch does
        let len = if b == 0 {
            t_pad
        } else {
            t_pad / 2 + (rng.next_u64() as usize) % (t_pad / 2 + 1)
        };
        for tok in &mut full_tokens[b * t_pad..][..len] {
            *tok = (rng.next_u64() % 50000) as u32;
        }
        actual_lens.push(len);
        audio_caps.push(300 + (rng.next_u64() as usize) % (N_AUDIO_CTX - 299));
    }
    DtwBatch {
        full_tokens,
        t_pad,
        actual_lens,
        audio_caps,
        slots: (0..rows).collect(),
    }
}

/// The DTW forward over cached cross keys and values, against the forward the reference
/// runs: the same rows with their keys and values projected over the rows stacked (a
/// fresh cache prefilled at that row count, which is that projection). The production
/// cache is prefilled in groups of other sizes, consecutive and not (the scratch path),
/// and the rows read it in runs that are consecutive, repeated, descending, strided over a
/// NaN slot and mixed. Raw scores and prepped matrices must match bit for bit and be
/// finite; reading a slot whose prefill isn't its current encoder output must fail
#[test]
fn dtw_cached_kv_matches_projection() -> Result<()> {
    let Some((gpu, blas)) = gpu_only()? else {
        return Ok(());
    };
    let Some(weights) = std::env::var_os("WHISPERDRZ_WEIGHTS").map(PathBuf::from) else {
        eprintln!("skipping: WHISPERDRZ_WEIGHTS not set (run scripts/gpu-test.sh)");
        return Ok(());
    };
    let mut w = Weights::load(&gpu, &weights, |n| n.starts_with("decoder."))?;
    let decoder = Decoder::new(&gpu, &mut w)?;
    let env = (&gpu, &blas, &decoder);

    const SLOTS: usize = 8;
    const POISONED: usize = 6;
    let mut rng = Rng::new(0xd7a_cac4e);
    // slot i's encoder output is window i
    let windows: Vec<Vec<u16>> = (0..SLOTS)
        .map(|_| rng.f16_normal(WINDOW, 0.0, 1.0))
        .collect();
    let stacked = |slots: &[usize]| -> Result<Buf> {
        gpu.upload(
            slots
                .iter()
                .flat_map(|&s| windows[s].clone())
                .collect::<Vec<u16>>(),
        )
    };

    let caches = KvCaches::new(&gpu, N_LAYER, SLOTS)?;
    for group in [&[0usize, 1, 2][..], &[5, 3], &[4], &[7]] {
        prefill(env, &caches, stacked(group)?.ptr(), group)?;
    }
    poison_slot(&gpu, &caches, POISONED)?;
    poison_slot(&gpu, &caches, caches.trash())?;

    let patterns: [&[usize]; 6] = [
        &[0, 1, 2],
        &[3, 3],
        &[5, 4, 3],
        &[5, 7],
        &[7, 0, 2, 4, 4],
        &[4],
    ];
    for (i, &slots) in patterns.iter().enumerate() {
        let t_pad = [61, 128, 200, 445, 97, 300][i];
        let batch = random_batch(&mut rng, slots, t_pad);
        let rows = slots.len();

        let reference = KvCaches::new(&gpu, N_LAYER, rows)?;
        let ref_slots: Vec<usize> = (0..rows).collect();
        prefill(env, &reference, stacked(slots)?.ptr(), &ref_slots)?;
        let want = decoder.dtw_scores(&gpu, &blas, &reference, &batch, &ref_slots)?;
        let got = decoder.dtw_scores(&gpu, &blas, &caches, &batch, slots)?;
        let shape = [ALIGNMENT_HEADS.len(), rows, t_pad, N_AUDIO_CTX];
        ensure!(
            want.shape() == shape && got.shape() == shape,
            "{slots:?}: scores {:?} and {:?}, want {shape:?}",
            want.shape(),
            got.shape()
        );
        let (want_raw, got_raw): (Vec<f32>, Vec<f32>) = (want.download(&gpu)?, got.download(&gpu)?);
        let n: usize = shape.iter().product();
        ensure!(
            want_raw.len() == n && got_raw.len() == n,
            "{slots:?}: {} and {} scores, want {n}",
            want_raw.len(),
            got_raw.len()
        );
        ensure!(
            got_raw.iter().all(|v| v.is_finite()),
            "{slots:?}: non-finite scores"
        );
        let differ = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count()
        };
        let d = differ(&got_raw, &want_raw);
        ensure!(
            d == 0,
            "{slots:?}: {d} of {n} scores differ from the projection"
        );

        // every row at full length runs the forward unpacked; the real tokens' scores
        // must not change (cuBLAS runs the GEMMs at another row count)
        let unpacked = DtwBatch {
            actual_lens: vec![t_pad; rows],
            ..batch.clone()
        };
        let padded = decoder
            .dtw_scores(&gpu, &blas, &caches, &unpacked, slots)?
            .download(&gpu)?;
        ensure!(
            padded.len() == n,
            "{slots:?}: {} padded scores",
            padded.len()
        );
        let planes = (rows, ALIGNMENT_HEADS.len(), t_pad);
        let inner = |v: &[f32]| heads_inner(v, (ALIGNMENT_HEADS.len(), rows, t_pad));
        let (pr, gr) = (
            real_tokens(&inner(&padded), planes, &batch.actual_lens),
            real_tokens(&inner(&got_raw), planes, &batch.actual_lens),
        );
        ensure!(
            pr.len() == gr.len() && !gr.is_empty(),
            "{slots:?}: {} and {} real-token scores",
            pr.len(),
            gr.len()
        );
        let (bits, max) = pr.iter().zip(&gr).fold((0, 0f64), |(d, m), (a, b)| {
            (
                d + usize::from(a.to_bits() != b.to_bits()),
                m.max((a - b).abs()),
            )
        });
        println!(
            "{slots:?} lens {:?}: packed vs padded forward, {bits} of {} real-token scores \
             differ, max {max:.3e}",
            batch.actual_lens,
            gr.len()
        );
        // a wrong row or token moves scores by their own size; GEMM rounding by far less
        let scale = pr.iter().fold(0f64, |m, v| m.max(v.abs()));
        ensure!(
            max <= 0.01 * scale,
            "{slots:?}: packed forward moves real-token scores by {max} (largest {scale})"
        );

        let (wm, gm) = (want.prep(&gpu, &batch)?, got.prep(&gpu, &batch)?);
        ensure!(
            gm.shape == wm.shape
                && gm.shape == [rows, t_pad, batch.max_audio_cap()]
                && gm.data.len() == gm.shape.iter().product::<usize>()
                && wm.data.len() == gm.data.len(),
            "{slots:?}: matrices {:?} ({} values) vs {:?} ({})",
            gm.shape,
            gm.data.len(),
            wm.shape,
            wm.data.len()
        );
        ensure!(
            gm.data.iter().all(|v| v.is_finite()),
            "{slots:?}: non-finite matrices"
        );
        let d = differ(&gm.data, &wm.data);
        ensure!(
            d == 0,
            "{slots:?}: {d} matrix values differ from the projection"
        );
        println!("{slots:?} T={t_pad}: {n} scores and the matrices bit for bit");
    }

    // a slot never prefilled, and one whose encoder output changed after its prefill
    let batch = random_batch(&mut rng, &[POISONED], 32);
    ensure!(
        decoder
            .dtw_scores(&gpu, &blas, &caches, &batch, &[POISONED])
            .is_err(),
        "DTW read slot {POISONED}, which no prefill filled"
    );
    let first = stacked(&[4])?;
    let stale = caches.store_xa(&gpu, first.ptr(), &[4])?;
    ensure!(
        decoder
            .dtw_scores(&gpu, &blas, &caches, &batch, &[4])
            .is_err(),
        "DTW read slot 4's cross keys after its encoder output was replaced"
    );

    // a source a later store replaced must not fill the cache: the check runs before the
    // GEMMs, so placeholder weights never reach cuBLAS
    let replacement = stacked(&[4])?;
    caches.store_xa(&gpu, replacement.ptr(), &[4])?;
    let w = first.ptr();
    ensure!(
        caches
            .fill_cross(&gpu, &blas, 7, &stale, (w, w, w), None)
            .is_err(),
        "fill_cross took an encoder output a later store_xa replaced"
    );
    ensure!(
        !caches.cross_from_xa(4, 1),
        "slot 4 claims cross keys from its current encoder output"
    );
    Ok(())
}

/// The DTW forward on another stream, as the transcriber queues it: a prefill of window
/// set A into slots, the forward on a second handle and a prefill of window set B into
/// the same slots, all queued with no host wait. The lease must order the forward after
/// A's keys land and B's prefill after the forward's reads, so the scores match the
/// forward on the caches' own stream bit for bit. Each round also queues B's encoder
/// output copy first thing, which `store_xa` fences too
#[test]
fn dtw_side_stream_is_fenced() -> Result<()> {
    let Some((gpu, blas)) = gpu_only()? else {
        return Ok(());
    };
    let Some(weights) = std::env::var_os("WHISPERDRZ_WEIGHTS").map(PathBuf::from) else {
        eprintln!("skipping: WHISPERDRZ_WEIGHTS not set (run scripts/gpu-test.sh)");
        return Ok(());
    };
    let mut w = Weights::load(&gpu, &weights, |n| n.starts_with("decoder."))?;
    let decoder = Decoder::new(&gpu, &mut w)?;
    let side = Gpu::with_priority(0, Priority::Mid)?;
    let side_blas = Blas::new(&side)?;

    let slots = [1usize, 2];
    let mut rng = Rng::new(0x51de_57a3);
    let window_set = |rng: &mut Rng| -> Result<Buf> {
        gpu.upload(rng.f16_normal(slots.len() * WINDOW, 0.0, 1.0))
    };
    let (a, b) = (window_set(&mut rng)?, window_set(&mut rng)?);
    let ws = Workspace::new(&gpu, 1, 1, slots.len())?;
    let batch = DtwBatch {
        slots: slots.to_vec(),
        ..random_batch(&mut rng, &slots, 300)
    };

    // A's scores on the caches' own stream
    let reference = KvCaches::new(&gpu, N_LAYER, 4)?;
    decoder.fill_cross(&gpu, &blas, &reference, &ws, a.ptr(), &slots)?;
    let want = decoder
        .dtw_scores(&gpu, &blas, &reference, &batch, &slots)?
        .download(&gpu)?;
    ensure!(
        want.iter().all(|v| v.is_finite()),
        "non-finite reference scores"
    );

    let caches = KvCaches::new(&gpu, N_LAYER, 4)?;
    decoder.fill_cross(&gpu, &blas, &caches, &ws, b.ptr(), &slots)?;
    gpu.synchronize()?;
    for round in 0..4 {
        decoder.fill_cross(&gpu, &blas, &caches, &ws, a.ptr(), &slots)?;
        let got = decoder.dtw_scores(&side, &side_blas, &caches, &batch, &slots)?;
        decoder.fill_cross(&gpu, &blas, &caches, &ws, b.ptr(), &slots)?;
        let got = got.download(&side)?;
        gpu.synchronize()?;
        ensure!(
            got.len() == want.len(),
            "round {round}: {} scores, want {}",
            got.len(),
            want.len()
        );
        let d = got
            .iter()
            .zip(&want)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        ensure!(
            d == 0,
            "round {round}: {d} of {} side-stream scores differ from the same-stream forward",
            want.len()
        );
    }

    println!(
        "side-stream DTW over slots {slots:?}: {} scores bit for bit in 4 rounds",
        want.len()
    );
    Ok(())
}
