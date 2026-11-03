//! DTW port replayed against the reference traces (`short3`, `e21c600`)
//!
//! Every recorded `_dtw_from_neg_matrix` and `_apply_anchor_bands` call is rerun on its
//! logged inputs, and every `fill_word_timings_batch` call is rerun end to end with the
//! GPU part answered by the recorded `_dtw_prep_batch` output. Each chunk's banded cost
//! matrix is also checked against the matrix the reference passed to
//! `_dtw_from_neg_matrix`, which covers the anchor bands even where the tracer logged no
//! `_apply_anchor_bands` call. Skips when the traces aren't there

// each test crate uses part of the shared helpers
#[allow(dead_code)]
#[path = "support/dtw.rs"]
mod support;

use std::cell::Cell;

use eyre::{Result, bail, ensure, eyre};
use serde_json::Value;
use support::{TableTokenizer, bits, opt_usize, strings, token_groups};
use whisperdrz_fixtures::{Array, Event, TraceSet, plain, read_npy};
use whisperdrz_native::dtw::{
    DtwOptions, Matrices, NegMatrix, PreparedBatch, TagAnchor, WordAnchor, apply_anchor_bands,
    dtw_from_neg_matrix, fill_word_timings_batch,
};

/// Calls checked and mismatches found for one label
#[derive(Default)]
struct Tally {
    checked: usize,
    mismatches: Vec<String>,
}

impl Tally {
    fn record(&mut self, what: String, outcome: Result<()>) {
        self.checked += 1;
        if let Err(e) = outcome {
            self.mismatches.push(format!("{what}: {e:#}"));
        }
    }

    fn report(&self, set: &str, label: &str) {
        eprintln!(
            "{set} {label}: {} checked, {} mismatched",
            self.checked,
            self.mismatches.len()
        );
        if let Some(first) = self.mismatches.first() {
            eprintln!("  first mismatch: {first}");
        }
    }
}

/// `TraceSet::array`, but also looking in the set's own `blobs/`, where the tracer
/// writes them (`TraceSet::blob_path` looks one directory up)
fn array(set: &TraceSet, v: &Value) -> Result<Array> {
    if let Some(sha) = v.get("blob").and_then(Value::as_str) {
        let path = set.dir().join("blobs").join(format!("{sha}.npy"));
        if path.exists() {
            return read_npy(&path);
        }
    }

    set.array(v)
}

fn matrix(set: &TraceSet, v: &Value) -> Result<NegMatrix> {
    let a = array(set, v)?;
    ensure!(a.shape.len() == 2, "matrix shape {:?}", a.shape);
    NegMatrix::new(a.shape[0], a.shape[1], a.to_f32()?)
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(Value::as_u64)
                .map(|x| x as usize)
                .collect()
        })
        .unwrap_or_default()
}

/// `(word, start, end)` triples from a recorded `_dtw_from_neg_matrix` return
fn recorded_times(ev: &Event) -> Result<Vec<(String, f64, f64)>> {
    plain(ev.ret()?)
        .as_array()
        .ok_or_else(|| eyre!("ret is not a list"))?
        .iter()
        .map(|t| {
            Ok((
                t[0].as_str().ok_or_else(|| eyre!("word"))?.to_owned(),
                t[1].as_f64().unwrap(),
                t[2].as_f64().unwrap(),
            ))
        })
        .collect()
}

fn check_neg_matrix_call(set: &TraceSet, ev: &Event) -> Result<()> {
    let neg = matrix(set, ev.arg("neg_matrix")?)?;
    let words = strings(ev.arg("external_words")?);
    let got = dtw_from_neg_matrix(
        &neg,
        words.len(),
        &token_groups(ev.arg("external_word_tokens")?),
    );
    let want = recorded_times(ev)?;
    ensure!(
        got.len() == want.len(),
        "{} words, recorded {}",
        got.len(),
        want.len()
    );
    for (i, (g, (w, s, e))) in got.iter().zip(&want).enumerate() {
        ensure!(
            words[i] == *w && g.start == *s && g.end == *e,
            "word {i} {w:?}: ({}, {}) vs recorded ({s}, {e})",
            g.start,
            g.end
        );
    }

    Ok(())
}

fn check_band_call(set: &TraceSet, ev: &Event) -> Result<()> {
    let mut neg = matrix(set, ev.arg("neg_matrix")?)?;
    let anchors = |name: &str| {
        plain(ev.arg(name).unwrap_or(&Value::Null))
            .as_array()
            .cloned()
            .unwrap_or_default()
    };
    let word_anchors: Vec<WordAnchor> = anchors("word_anchors")
        .iter()
        .map(|a| WordAnchor {
            word: a[0].as_u64().unwrap() as usize,
            start: a[1].as_f64().unwrap(),
            end: a[2].as_f64().unwrap(),
        })
        .collect();
    let tag_anchors: Vec<TagAnchor> = anchors("tag_anchors")
        .iter()
        .map(|a| TagAnchor {
            start: a[0].as_f64().unwrap(),
            end: a[1].as_f64().unwrap(),
            prev: opt_usize(&a[2]),
            next: opt_usize(&a[3]),
        })
        .collect();
    apply_anchor_bands(
        &mut neg,
        strings(ev.arg("external_words")?).len(),
        &token_groups(ev.arg("external_word_tokens")?),
        ev.arg("audio_cap")?.as_u64().unwrap() as usize,
        &word_anchors,
        &tag_anchors,
        ev.arg("slack_frames")?.as_i64().unwrap(),
    )?;
    let after = matrix(set, ev.after_arg("neg_matrix")?)?;
    ensure!(
        bits(&neg.data) == bits(&after.data),
        "banded matrix differs"
    );
    Ok(())
}

/// One `fill_word_timings_batch` call and the GPU-side events recorded inside it
struct BatchCall<'e> {
    fill: &'e Event,
    prep: Option<&'e Event>,
    forward: Option<&'e Event>,
    dtw: Vec<&'e Event>,
}

fn batch_calls(events: &[Event]) -> Vec<BatchCall<'_>> {
    // a call's nested events are emitted before it (seq is taken when a call returns)
    let mut out = Vec::new();
    let mut window: Vec<&Event> = Vec::new();
    for ev in events {
        match ev.ev.as_str() {
            "fill_word_timings_batch" => {
                let last = |label: &str| window.iter().rev().find(|e| e.ev == label).copied();
                out.push(BatchCall {
                    fill: ev,
                    prep: last("_dtw_prep_batch"),
                    forward: last("dtw_decoder_forward"),
                    dtw: window
                        .iter()
                        .filter(|e| e.ev == "_dtw_from_neg_matrix")
                        .copied()
                        .collect(),
                });
                window.clear();
            }
            "_dtw_prep_batch" | "dtw_decoder_forward" | "_dtw_from_neg_matrix" => window.push(ev),
            _ => {}
        }
    }

    out
}

fn slots(fill: &Event) -> Result<Vec<(String, i64)>> {
    fill.arg("slot_data")?
        .as_array()
        .ok_or_else(|| eyre!("slot_data is not a list"))?
        .iter()
        .map(|s| {
            let t = &s["__tuple"];
            Ok((
                t[1].as_str().ok_or_else(|| eyre!("slot text"))?.to_owned(),
                t[2].as_i64().ok_or_else(|| eyre!("num_frames"))?,
            ))
        })
        .collect()
}

fn recorded_matrices(set: &TraceSet, prep: &Event) -> Result<Matrices> {
    let a = array(set, prep.ret()?)?;
    ensure!(a.shape.len() == 3, "prep shape {:?}", a.shape);
    Ok(Matrices {
        shape: [a.shape[0], a.shape[1], a.shape[2]],
        data: a.to_f32()?,
    })
}

/// Reruns the call end to end, checking the batch against the recorded GPU inputs
fn check_batch_end_to_end(
    set: &TraceSet,
    tok: &TableTokenizer,
    call: &BatchCall<'_>,
) -> Result<()> {
    let slots = slots(call.fill)?;
    let called = Cell::new(false);
    let got = fill_word_timings_batch(&slots, tok, &DtwOptions::default(), |batch| {
        called.set(true);
        let prep = call
            .prep
            .ok_or_else(|| eyre!("the reference ran no forward"))?;
        ensure!(
            batch.actual_lens == usizes(prep.arg("actual_lens")?),
            "actual_lens {:?} vs recorded {}",
            batch.actual_lens,
            prep.arg("actual_lens")?
        );
        ensure!(
            batch.audio_caps == usizes(prep.arg("audio_caps")?),
            "audio_caps {:?} vs recorded {}",
            batch.audio_caps,
            prep.arg("audio_caps")?
        );
        if let Some(forward) = call.forward {
            let x = array(set, forward.arg("x")?)?;
            let want = x.to_i64()?;
            let have: Vec<i64> = batch.full_tokens.iter().map(|&t| t as i64).collect();
            ensure!(
                x.shape == [batch.rows(), batch.t_pad] && want == have,
                "full_tokens differ from the recorded forward"
            );
        }

        recorded_matrices(set, prep)
    })?;
    ensure!(
        called.get() == call.prep.is_some(),
        "forward ran in only one of Python and Rust"
    );

    let want = strings(call.fill.ret()?);
    for (si, (g, w)) in got.iter().zip(&want).enumerate() {
        ensure!(
            g == w,
            "slot {si} text differs\n  rust:   {g:?}\n  python: {w:?}"
        );
    }

    ensure!(
        got.len() == want.len(),
        "{} texts, recorded {}",
        got.len(),
        want.len()
    );
    Ok(())
}

/// Checks each chunk's banded cost matrix against the one the reference passed to
/// `_dtw_from_neg_matrix`, and that every recorded call is accounted for
fn check_batch_chunks(
    set: &TraceSet,
    tok: &TableTokenizer,
    call: &BatchCall<'_>,
    chunks: &mut usize,
) -> Result<()> {
    let slots = slots(call.fill)?;
    let prepared = PreparedBatch::new(&slots, tok, &DtwOptions::default())?;
    let Some(batch) = prepared.batch() else {
        ensure!(
            call.dtw.is_empty(),
            "no chunks but {} recorded DTW calls",
            call.dtw.len()
        );
        return Ok(());
    };

    let matrices = recorded_matrices(set, call.prep.ok_or_else(|| eyre!("no recorded prep"))?)?;
    let mut used = vec![false; call.dtw.len()];
    for bi in 0..batch.rows() {
        let neg = prepared.chunk_neg_matrix(bi, &matrices)?;
        if neg.is_empty() {
            continue;
        }

        let rec = prepared.chunk(bi);
        let mut found = false;
        for (k, ev) in call.dtw.iter().enumerate() {
            if used[k]
                || strings(ev.arg("external_words")?) != rec.ext_words
                || token_groups(ev.arg("external_word_tokens")?) != rec.ext_word_tokens
            {
                continue;
            }

            let want = matrix(set, ev.arg("neg_matrix")?)?;
            if (want.rows, want.cols) == (neg.rows, neg.cols) && bits(&want.data) == bits(&neg.data)
            {
                used[k] = true;
                found = true;
                break;
            }

            bail!(
                "chunk {bi} ({} words) cost matrix differs from recorded call #{}",
                rec.ext_words.len(),
                ev.seq
            );
        }

        ensure!(found, "chunk {bi} has no recorded DTW call with its words");
        *chunks += 1;
    }

    ensure!(
        used.iter().all(|&u| u),
        "{} recorded DTW calls matched no chunk",
        used.iter().filter(|&&u| !u).count()
    );
    Ok(())
}

fn replay(name: &str) {
    let Some(set) = TraceSet::open(name) else {
        return;
    };
    let tok = TableTokenizer::load();
    let (mut neg_calls, mut band_calls, mut batches, mut chunk_mats) = (
        Tally::default(),
        Tally::default(),
        Tally::default(),
        Tally::default(),
    );
    let mut chunks = 0;

    for stem in set.stems().unwrap() {
        let events = set.events(&stem).unwrap();
        for ev in &events {
            match ev.ev.as_str() {
                "_dtw_from_neg_matrix" => neg_calls.record(
                    format!("{stem} #{}", ev.seq),
                    check_neg_matrix_call(&set, ev),
                ),
                "_apply_anchor_bands" => {
                    band_calls.record(format!("{stem} #{}", ev.seq), check_band_call(&set, ev))
                }
                _ => {}
            }
        }

        for call in batch_calls(&events) {
            let what = format!("{stem} #{}", call.fill.seq);
            if call.fill.raise.is_some() {
                batches.record(what, Err(eyre!("the reference raised; not replayed")));
                continue;
            }

            batches.record(what.clone(), check_batch_end_to_end(&set, &tok, &call));
            chunk_mats.record(what, check_batch_chunks(&set, &tok, &call, &mut chunks));
        }
    }

    neg_calls.report(name, "_dtw_from_neg_matrix");
    band_calls.report(name, "_apply_anchor_bands");
    batches.report(name, "fill_word_timings_batch end to end");
    chunk_mats.report(
        name,
        &format!("fill_word_timings_batch banded chunk matrices ({chunks} chunks)"),
    );

    for t in [&neg_calls, &band_calls, &batches, &chunk_mats] {
        assert!(t.mismatches.is_empty(), "{name}: {}", t.mismatches[0]);
    }

    assert!(
        neg_calls.checked > 0 && batches.checked > 0,
        "{name}: no DTW calls in the trace"
    );
}

#[test]
fn replay_short3() {
    replay("short3");
}

#[test]
fn replay_e21c600() {
    replay("e21c600");
}
