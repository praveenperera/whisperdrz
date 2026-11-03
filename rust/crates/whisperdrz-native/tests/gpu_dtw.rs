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
//! start or end moves and by how many 0.02 s frames. Skips unless `WHISPERDRZ_WEIGHTS` and
//! `WHISPERDRZ_DECODER_REF` are set; run through the GPU queue with
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
use numerics::Err;
use parity::{Check, MAX_RATIO, Row, f16_bits, f64s, report};
use serde_json::Value;
use whisperdrz_fixtures::{ArrayData, Event, TraceSet};
use whisperdrz_native::cuda::decoder::{ALIGNMENT_HEADS, DtwCapture, DtwScores, N_AUDIO_CTX};
use whisperdrz_native::cuda::{Blas, Buf, Decoder, Gpu, Weights};
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
        let ptrs: Vec<u64> = (0..rows)
            .map(|i| xa.at(i * N_AUDIO_CTX * 1280 * 2))
            .collect();
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
        let scores = decoder.dtw_scores(gpu, blas, &batch, &ptrs)?;
        ensure!(
            scores.shape() == [heads, rows, t, N_AUDIO_CTX],
            "{name}: our scores {:?}",
            scores.shape()
        );
        let ours = heads_inner(&scores.download(gpu)?, (heads, rows, t));
        let torch16 = f64v(&torch_scores);
        let same = ours.iter().zip(&torch16).filter(|(a, b)| a == b).count();
        let truth = f64s(&refs.join(case["scores32"].as_str().unwrap_or_default()))?;
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
        let ours_m = f64v(&m.data);
        let direct = Err::of(&ours_m, &f64v(&recorded_m))?;
        let truth_m = f64s(&refs.join(case["matrices32"].as_str().unwrap_or_default()))?;
        matrix_rows.push(variant_bar(
            Row::new(&name, &ours_m, &f64v(&recorded_m), &truth_m)?,
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
                capture.gather(gpu, s, (qd.ptr(), kd.ptr()), head)?;
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
            let ptrs: Vec<u64> = batch.slots.iter().map(|&si| xa[si].ptr()).collect();
            let ours = decoder.dtw_prep(gpu, blas, batch, &ptrs)?;
            prepared.check_matrices(&ours)?;
            let rec = set.array(prep.ret()?)?;
            let recorded = Matrices {
                shape: [rec.shape[0], rec.shape[1], rec.shape[2]],
                data: rec.to_f32()?,
            };

            let (mut c_words, mut c_moved, mut c_max, mut c_frames) = (0, 0, 0i64, 0i64);
            let mut results = Vec::new();
            for bi in 0..batch.rows() {
                let (neg_o, neg_r) = (
                    prepared.chunk_neg_matrix(bi, &ours)?,
                    prepared.chunk_neg_matrix(bi, &recorded)?,
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
