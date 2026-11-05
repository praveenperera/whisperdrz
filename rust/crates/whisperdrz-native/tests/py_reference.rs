//! Checks the ports against values the Python reference computed on the CPU
//! (`rust/tools/gen_post_data.py`): CPython `random` draws, torch CPU f32 kernels and
//! clustering, and every text function on edge cases and a seeded fuzz corpus

use std::path::PathBuf;

use eyre::{Result, ensure, eyre};
use serde_json::{Value, json};
use whisperdrz_native::post::{self, FinalizeState, py};
use whisperdrz_native::speaker::{PyRandom, cluster_speakers, torch_cpu};

fn data(name: &str) -> Result<Value> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    Ok(serde_json::from_str(&std::fs::read_to_string(&path)?)?)
}

fn f32s(v: &Value) -> Vec<f32> {
    v.as_array()
        .into_iter()
        .flatten()
        .map(|b| f32::from_bits(b.as_u64().unwrap_or(0) as u32))
        .collect()
}

fn same_bits(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
        + a.len().abs_diff(b.len())
}

#[test]
fn pyrandom_matches_cpython() -> Result<()> {
    for case in data("pyrandom.json")?
        .as_array()
        .ok_or_else(|| eyre!("not a list"))?
    {
        let seed = case["seed"].as_u64().ok_or_else(|| eyre!("seed"))?;
        let mut r = PyRandom::new(seed);
        for (i, call) in case["calls"].as_array().into_iter().flatten().enumerate() {
            let args = call[1].as_array().ok_or_else(|| eyre!("args"))?;
            let got = match call[0].as_str().unwrap_or_default() {
                "random" => json!(r.random()),
                "uniform" => json!(r.uniform(
                    args[0].as_f64().unwrap_or(0.0),
                    args[1].as_f64().unwrap_or(0.0)
                )),
                "getrandbits" => json!(r.getrandbits(args[0].as_u64().unwrap_or(0) as u32)),
                "randint" => {
                    json!(r.randint(args[0].as_i64().unwrap_or(0), args[1].as_i64().unwrap_or(0))?)
                }
                other => return Err(eyre!("unknown op {other}")),
            };
            ensure!(got == call[2], "seed {seed} draw {i} {call}: got {got}");
        }
    }

    Ok(())
}

#[test]
fn torch_kernels_match_bits() -> Result<()> {
    let d = data("torch_cpu.json")?;
    let x = f32s(&d["x"]);
    let mut xn = x.clone();
    for row in xn.as_chunks_mut::<192>().0 {
        torch_cpu::normalize(row);
    }

    ensure!(
        same_bits(&xn, &f32s(&d["normalize"])) == 0,
        "normalize differs"
    );

    let want = f32s(&d["normalize"]);
    let sumsq: Vec<f32> = want
        .as_chunks::<192>()
        .0
        .iter()
        .map(|r| torch_cpu::sum(&r.map(|v| v * v)))
        .collect();
    ensure!(
        same_bits(&sumsq, &f32s(&d["sumsq"])) == 0,
        "pow(2).sum(-1) differs"
    );

    let cd = torch_cpu::cdist(&want[..25 * 192], &want[..25 * 192], 192);
    ensure!(
        same_bits(&cd, &f32s(&d["cdist25"])) == 0,
        "cdist (25 rows) differs"
    );
    let cd = torch_cpu::cdist(&want[..7 * 192], &want[30 * 192..33 * 192], 192);
    ensure!(
        same_bits(&cd, &f32s(&d["cdist7x3"])) == 0,
        "cdist (7x3) differs"
    );

    let cols: Vec<f32> = (0..192)
        .map(|c| torch_cpu::column_sum(&(0..40).map(|r| x[r * 192 + c]).collect::<Vec<_>>()))
        .collect();
    ensure!(
        same_bits(&cols, &f32s(&d["colsum40"])) == 0,
        "sum(0) over 40 rows differs"
    );
    Ok(())
}

#[test]
fn clustering_matches_torch() -> Result<()> {
    let d = data("torch_cpu.json")?;
    for (i, case) in d["clusters"].as_array().into_iter().flatten().enumerate() {
        let emb: Vec<Vec<f32>> = f32s(&case["emb"])
            .as_chunks::<192>()
            .0
            .iter()
            .map(|r| r.to_vec())
            .collect();
        let threshold = case["threshold"].as_f64().unwrap_or(0.0);
        let min_size = case["min_cluster_size"].as_u64().unwrap_or(1) as usize;
        let c = cluster_speakers(&emb, threshold, min_size)?;
        let want: Vec<i64> = case["labels"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_i64)
            .collect();
        ensure!(
            c.labels == want,
            "case {i} (n={}): want {want:?}, got {:?}",
            emb.len(),
            c.labels
        );
    }

    Ok(())
}

fn segments(v: &[post::Segment]) -> Value {
    serde_json::to_value(v).unwrap_or(Value::Null)
}

fn run(fn_name: &str, args: &[Value]) -> Result<Value> {
    let text = || args[0].as_str().ok_or_else(|| eyre!("text arg"));
    let num = |i: usize| args[i].as_f64().ok_or_else(|| eyre!("float arg"));
    Ok(match fn_name {
        "cut_after_last_timed_word" => json!(post::cut_after_last_timed_word(text()?)?),
        "fix_nonmonotonic_timestamps" => json!(post::fix_nonmonotonic_timestamps(text()?)?),
        "fix_speaker_ids" => json!(post::fix_speaker_ids(text()?)?),
        "merge_consecutive_speakers" => json!(post::merge_consecutive_speakers(text()?)?),
        "remove_speaker_tags_not_after_newline" => {
            json!(post::remove_speaker_tags_not_after_newline(text()?)?)
        }
        "consolidate_adjacent_tags" => json!(post::consolidate_adjacent_tags(text()?)?),
        "remove_orphan_paired_tags" => json!(post::remove_orphan_paired_tags(text()?)?),
        "remove_invalid_timing_pairs" => json!(post::remove_invalid_timing_pairs(text()?, 0.02)?),
        "remove_short_nonspeech" => json!(post::remove_short_nonspeech(text()?, 0.02)?),
        "remove_short_pauses" => json!(post::remove_short_pauses(text()?, 800)?),
        "normalize_speaker_ids" => json!(post::normalize_speaker_ids(text()?)?),
        "strip_to_last_closing" => json!(post::strip_to_last_closing(text()?)?),
        "parse_to_segments" => segments(&post::parse_to_segments(text()?)?),
        "reconcile_decoder_and_alignment_timings" => {
            json!(post::reconcile_decoder_and_alignment_timings(
                text()?,
                args[1].as_i64().unwrap_or(92)
            )?)
        }
        "clean_chunk" => {
            let c = post::clean_chunk(text()?, args[1].as_i64().unwrap_or(92))?;
            json!([c.text, c.speaker_order])
        }
        "quantize_timestamps" => json!(post::quantize_timestamps(text()?, num(1)?, 0.02)?),
        "offset_timestamps" => json!(post::offset_timestamps(text()?, num(1)?)?),
        "finalize_transcription" => {
            let st = FinalizeState {
                offset: args[0]["offset"].as_f64().unwrap_or(0.0),
                keepspeaker: false,
                cutearly: args[0]["cutearly"].as_bool().unwrap_or(false),
            };
            let cur = args[1].as_str().ok_or_else(|| eyre!("cur arg"))?;
            json!(post::finalize_transcription(
                &st,
                cur,
                args[2].as_bool().unwrap_or(true)
            )?)
        }
        "fmt_2f" => json!(py::fmt_2f(num(0)?)),
        "round2" => json!(py::round2(num(0)?)),
        "round_to" => json!(post::round_to(num(0)?, num(1)?)),
        other => return Err(eyre!("unknown function {other}")),
    })
}

#[test]
fn text_functions_match_python() -> Result<()> {
    let cases = data("text_cases.json")?;
    let cases = cases.as_array().ok_or_else(|| eyre!("not a list"))?;
    let mut bad = Vec::new();
    for case in cases {
        let name = case["fn"].as_str().unwrap_or_default();
        let args = case["args"].as_array().ok_or_else(|| eyre!("args"))?;
        let got = run(name, args);
        let ok = match (&got, case.get("ret")) {
            (Ok(v), Some(want)) => v == want,
            // Python raised; the port must fail too
            (Err(_), None) => true,
            _ => false,
        };
        if !ok {
            bad.push(format!("{name}({args:?}): want {case}, got {got:?}"));
        }
    }

    ensure!(
        bad.is_empty(),
        "{} of {} cases differ; first: {}",
        bad.len(),
        cases.len(),
        bad[0]
    );
    Ok(())
}
