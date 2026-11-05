//! Replays every recorded call of the post-processing and speaker-remap functions
//! from the reference traces and compares return values (and, where recorded, the
//! arguments after the call) with the Rust ports. Skips when the traces are missing

use std::collections::{BTreeMap, VecDeque};

use eyre::{Result, bail, ensure, eyre};
use serde_json::{Value, json};
use whisperdrz_fixtures::{Event, TraceSet, as_f64};
use whisperdrz_native::post::{self, FinalizeState, Segment};
use whisperdrz_native::speaker::{self, CropRng, EMBED_DIM, PyRandom};

/// Per-label tallies and the first mismatch of each
#[derive(Default)]
struct Tally {
    calls: BTreeMap<String, (usize, usize)>,
    first: BTreeMap<String, String>,
    skipped: BTreeMap<String, usize>,
}

impl Tally {
    fn check(&mut self, label: &str, ok: bool, detail: impl FnOnce() -> String) {
        let e = self.calls.entry(label.to_owned()).or_default();
        e.0 += 1;
        if !ok {
            e.1 += 1;
            self.first.entry(label.to_owned()).or_insert_with(detail);
        }
    }

    fn skip(&mut self, label: &str) {
        *self.skipped.entry(label.to_owned()).or_default() += 1;
    }

    fn report(&self, name: &str) -> usize {
        eprintln!("== {name}");
        let mut bad = 0;
        for (label, (n, m)) in &self.calls {
            eprintln!("{label:60} {n:6} calls {m:4} mismatches");
            bad += m;
        }

        for (label, n) in &self.skipped {
            eprintln!("{label:60} {n:6} skipped (not replayable)");
        }

        for (label, d) in &self.first {
            eprintln!("first mismatch in {label}: {d}");
        }

        bad
    }
}

fn s<'a>(ev: &'a Event, name: &str) -> Result<&'a str> {
    ev.arg(name)?
        .as_str()
        .ok_or_else(|| eyre!("{} arg {name} is not a string", ev.ev))
}

fn f_or(ev: &Event, name: &str, default: f64) -> Result<f64> {
    match ev.args.get(name) {
        Some(v) => as_f64(v).ok_or_else(|| eyre!("{} arg {name} is not a float", ev.ev)),
        None => Ok(default),
    }
}

fn i_or(ev: &Event, name: &str, default: i64) -> Result<i64> {
    match ev.args.get(name) {
        Some(v) => v
            .as_i64()
            .ok_or_else(|| eyre!("{} arg {name} is not an int", ev.ev)),
        None => Ok(default),
    }
}

fn str_ret(ev: &Event) -> Result<&str> {
    ev.ret()?
        .as_str()
        .ok_or_else(|| eyre!("{} ret is not a string", ev.ev))
}

fn tuple(v: &Value) -> Result<&Vec<Value>> {
    v.get("__tuple")
        .and_then(Value::as_array)
        .ok_or_else(|| eyre!("not a tuple: {v}"))
}

fn audio_ref(v: &Value) -> Result<(usize, usize)> {
    let a = v
        .get("__audio")
        .and_then(Value::as_array)
        .ok_or_else(|| eyre!("not an audio ref: {v}"))?;
    Ok((
        a[0].as_u64().unwrap_or(0) as usize,
        a[1].as_u64().unwrap_or(0) as usize,
    ))
}

fn embedding(v: &Value) -> Result<Vec<f32>> {
    let data = v
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| eyre!("embedding not inline"))?;
    data.iter()
        .map(|x| {
            as_f64(x)
                .map(|x| x as f32)
                .ok_or_else(|| eyre!("bad float"))
        })
        .collect()
}

fn segments_json(segs: &[Segment]) -> Value {
    serde_json::to_value(segs).unwrap_or(Value::Null)
}

/// A crop RNG that draws from CPython's generator and checks each draw against the
/// recorded `random.*` events
struct CheckedRng<'a> {
    rng: PyRandom,
    expected: &'a mut VecDeque<Event>,
    bad: Vec<String>,
}

impl CheckedRng<'_> {
    fn expect(&mut self, name: &str, args: Value, got: Value) {
        match self.expected.pop_front() {
            Some(ev) => {
                let want_args = ev.args.get("args").cloned().unwrap_or(Value::Null);
                let want = ev.ret.clone().unwrap_or(Value::Null);
                if ev.ev != format!("random.{name}") || want_args != args || want != got {
                    self.bad.push(format!(
                        "#{} {} {want_args} -> {want}; got {name} {args} -> {got}",
                        ev.seq, ev.ev
                    ));
                }
            }
            None => self
                .bad
                .push(format!("unrecorded draw {name} {args} -> {got}")),
        }
    }
}

impl CropRng for CheckedRng<'_> {
    fn uniform(&mut self, a: f64, b: f64) -> f64 {
        let v = self.rng.uniform(a, b);
        self.expect("uniform", json!([a, b]), json!(v));
        v
    }

    fn randint(&mut self, a: i64, b: i64) -> Result<i64> {
        let v = self.rng.randint(a, b)?;
        self.expect("randint", json!([a, b]), json!(v));
        Ok(v)
    }
}

fn replay_text(ev: &Event, t: &mut Tally) -> Result<bool> {
    let label = ev.ev.as_str();
    let got: Value = match label {
        "clean.timings.parse_timing" => json!(post::parse_timing(s(ev, "t")?)?),
        "clean.timings.round_to" => json!(post::round_to(
            f_or(ev, "x", f64::NAN)?,
            f_or(ev, "step", f64::NAN)?
        )),
        "clean.timings.cut_after_last_timed_word" => {
            json!(post::cut_after_last_timed_word(s(ev, "input_string")?)?)
        }
        "clean.timings.quantize_timestamps" => {
            json!(post::quantize_timestamps(
                s(ev, "text")?,
                f_or(ev, "offset", 0.0)?,
                f_or(ev, "step", 0.02)?
            )?)
        }
        "clean.timings.offset_timestamps" => {
            json!(post::offset_timestamps(
                s(ev, "text")?,
                f_or(ev, "offset", f64::NAN)?
            )?)
        }
        "clean.timings.reconcile_decoder_and_alignment_timings" => {
            json!(post::reconcile_decoder_and_alignment_timings(
                s(ev, "text")?,
                i_or(ev, "confidence_threshold", 92)?
            )?)
        }
        "clean.timings.remove_invalid_timing_pairs" => {
            json!(post::remove_invalid_timing_pairs(
                s(ev, "text")?,
                f_or(ev, "min_duration", 0.02)?
            )?)
        }
        "clean.timings.fix_nonmonotonic_timestamps" => {
            json!(post::fix_nonmonotonic_timestamps(s(ev, "text")?)?)
        }
        "clean.timings.remove_short_nonspeech" => {
            json!(post::remove_short_nonspeech(
                s(ev, "text")?,
                f_or(ev, "min_duration_s", 0.02)?
            )?)
        }
        "clean.timings.remove_short_pauses" => {
            json!(post::remove_short_pauses(
                s(ev, "text")?,
                i_or(ev, "min_pause_ms", 800)?
            )?)
        }
        "clean.speakers.fix_speaker_ids" => json!(post::fix_speaker_ids(s(ev, "text")?)?),
        "clean.speakers.remove_speaker_tags_not_after_newline" => {
            json!(post::remove_speaker_tags_not_after_newline(s(ev, "text")?)?)
        }
        "clean.speakers.merge_consecutive_speakers" => {
            json!(post::merge_consecutive_speakers(s(ev, "text")?)?)
        }
        "clean.tags.consolidate_adjacent_tags" => {
            json!(post::consolidate_adjacent_tags(s(ev, "text")?)?)
        }
        "clean.tags.remove_orphan_paired_tags" => {
            json!(post::remove_orphan_paired_tags(s(ev, "text")?)?)
        }
        "speaker_clustering.normalize_speaker_ids" => {
            json!(post::normalize_speaker_ids(s(ev, "text")?)?)
        }
        "_strip_to_last_closing" => json!(post::strip_to_last_closing(s(ev, "cur")?)?),
        "parse_to_segments" => segments_json(&post::parse_to_segments(s(ev, "text")?)?),
        "inference_utils.finalize_transcription" => {
            let st = ev
                .arg("state")?
                .get("__state")
                .ok_or_else(|| eyre!("state not a snapshot"))?;
            let state = FinalizeState {
                offset: st.get("offset").and_then(as_f64).unwrap_or(0.0),
                keepspeaker: st
                    .get("keepspeaker")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                cutearly: st.get("cutearly").and_then(Value::as_bool).unwrap_or(false),
            };
            let apply = ev
                .args
                .get("apply_offset")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            json!(post::finalize_transcription(&state, s(ev, "cur")?, apply)?)
        }
        "clean.timings.quantize_timestamp" => {
            // the `re.Match` argument isn't serialized; `quantize_timestamps` covers it
            t.skip(label);
            return Ok(true);
        }
        _ => return Ok(false),
    };

    let want = ev.ret()?;
    t.check(label, &got == want, || {
        format!("#{}: want {want}, got {got}", ev.seq)
    });
    if let Some(after) = &ev.after {
        // none of these mutate their arguments; the recorded after-values confirm it
        let same = after.iter().all(|(k, v)| ev.args.get(k) == Some(v));
        t.check(&format!("{label} (after)"), same, || {
            format!("#{} mutated its arguments", ev.seq)
        });
    }

    Ok(true)
}

fn replay_cluster(ev: &Event, t: &mut Tally, margins: &mut Vec<String>) -> Result<()> {
    let embs = ev
        .arg("embeddings")?
        .as_array()
        .ok_or_else(|| eyre!("embeddings not a list"))?;
    let mut keys = Vec::new();
    let mut vecs = Vec::new();
    for e in embs {
        let tup = tuple(e)?;
        keys.push((tup[0].clone(), tup[1].clone()));
        vecs.push(embedding(&tup[2])?);
    }

    let threshold = f_or(ev, "threshold", 0.6)?;
    let min_size = i_or(ev, "min_cluster_size", 1)? as usize;
    let c = speaker::cluster_speakers(&vecs, threshold, min_size)?;
    let got: Vec<Value> = keys
        .iter()
        .zip(&c.labels)
        .map(|((a, b), l)| json!([{"__tuple": [a, b]}, l]))
        .collect();
    let want = ev.ret()?.get("__items").cloned().unwrap_or(Value::Null);
    let got = Value::Array(got);
    t.check(&ev.ev, got == want, || {
        format!("#{}: want {want}, got {got}", ev.seq)
    });

    let t32 = threshold as f32;
    let tightest = c
        .steps
        .iter()
        .map(|s| ((s.runner_up - s.min).min((s.min - t32).abs()), s))
        .min_by(|a, b| a.0.total_cmp(&b.0));
    margins.push(format!(
        "n={} steps={} matmul_steps={} tightest={:?} matmul_margin={:?}",
        vecs.len(),
        c.steps.len(),
        c.steps.iter().filter(|s| s.matmul).count(),
        tightest,
        c.matmul_margin(threshold)
    ));
    Ok(())
}

/// The text side of `_finalize_slot_after_alignment`: the window append and, for a
/// finished chunk, the cleanup chain and chunk offset
fn replay_slot(ev: &Event, t: &mut Tally, align_override_score: i64) -> Result<()> {
    let label = "_finalize_slot_after_alignment (text)";
    let st = ev
        .arg("s")?
        .get("__state")
        .ok_or_else(|| eyre!("s not a snapshot"))?;
    let after = ev
        .after_arg("s")?
        .get("__state")
        .ok_or_else(|| eyre!("after s not a snapshot"))?;
    let Some(pre) = ev.args.get("pre_augmented_cur").and_then(Value::as_str) else {
        // the window text comes from the DTW fill, which this module doesn't own
        t.skip(label);
        return Ok(());
    };

    let phase = s(ev, "phase")?;
    let more = ev.arg("more")?.as_bool().unwrap_or(false);
    let mut sofar = st["sofar"].as_str().unwrap_or_default().to_owned();
    if phase == "process" {
        let cur = if more {
            post::strip_to_last_closing(pre)?
        } else {
            pre.to_owned()
        };
        post::append_window(&mut sofar, &cur, true);
    } else if st["status"] == "finished" {
        post::append_window(&mut sofar, pre, false);
    }

    let want_sofar = after["sofar"].as_str().unwrap_or_default();
    t.check(&format!("{label} sofar"), sofar == want_sofar, || {
        format!("#{}: want {want_sofar:?}, got {sofar:?}", ev.seq)
    });

    let ret = ev.ret()?;
    if ret.is_null() {
        return Ok(());
    }

    let cleaned = post::clean_chunk(&sofar, align_override_score)?;
    let offset = st["meta"].get("offset").and_then(as_f64).unwrap_or(0.0);
    let text = if offset > 0.0 {
        post::offset_timestamps(&cleaned.text, offset)?
    } else {
        cleaned.text.clone()
    };
    let want = ret["prediction"].as_str().unwrap_or_default();
    t.check(&format!("{label} prediction"), text == want, || {
        format!("#{}: want {want:?}, got {text:?}", ev.seq)
    });

    let map: Vec<Value> = cleaned
        .speaker_order
        .iter()
        .enumerate()
        .map(|(i, o)| json!([i, o]))
        .collect();
    // the tracer writes an empty dict as `{}` (all of its zero keys are strings)
    let want_map = match &ret["speaker_id_map"] {
        Value::Object(o) if o.is_empty() => json!([]),
        m => m["__items"].clone(),
    };
    t.check(
        &format!("{label} speaker_id_map"),
        Value::Array(map.clone()) == want_map,
        || format!("#{}: want {want_map}, got {map:?}", ev.seq),
    );
    Ok(())
}

fn replay_file(set: &TraceSet, stem: &str, t: &mut Tally, margins: &mut Vec<String>) -> Result<()> {
    let events = set.events(stem)?;
    let mut randoms: VecDeque<Event> = events
        .iter()
        .filter(|e| e.ev.starts_with("random."))
        .cloned()
        .collect();
    let embeds: Vec<&Event> = events
        .iter()
        .filter(|e| e.ev == "_embed_audio_segment")
        .collect();
    let score = events
        .iter()
        .find(|e| e.ev == "clean.timings.reconcile_decoder_and_alignment_timings")
        .and_then(|e| e.args.get("confidence_threshold")?.as_i64())
        .unwrap_or(92);
    let mut remap_text: Option<String> = None;

    for ev in &events {
        if replay_text(ev, t)? {
            continue;
        }

        match ev.ev.as_str() {
            "speaker_clustering.cluster_speakers" => replay_cluster(ev, t, margins)?,
            "_finalize_slot_after_alignment" => replay_slot(ev, t, score)?,
            "extract_line_audio_segments" => {
                let (_, audio_len) = audio_ref(ev.arg("audio")?)?;
                // the only consumer of `random` in a traced file: a fresh seed(0) state
                let mut q = randoms.clone();
                let mut rng = CheckedRng {
                    rng: PyRandom::new(0),
                    expected: &mut q,
                    bad: Vec::new(),
                };
                let crops = speaker::extract_line_audio_segments(
                    audio_len,
                    s(ev, "text")?,
                    f_or(ev, "min_duration", 0.3)?,
                    f_or(ev, "max_duration", 10.0)?,
                    &mut rng,
                )?;
                let bad = std::mem::take(&mut rng.bad);
                t.check("random (sequence)", bad.is_empty(), || bad.join("; "));
                let got: Vec<Value> = crops
                    .iter()
                    .map(|c| json!({"__tuple": [c.line, c.speaker, {"__audio": [c.start, c.len]}]}))
                    .collect();
                let got = Value::Array(got);
                let want = ev.ret()?;
                t.check(&ev.ev, &got == want, || {
                    format!("#{}: want {want}, got {got}", ev.seq)
                });
            }
            "remap_speakers_per_line" => {
                let (_, audio_len) = audio_ref(ev.arg("audio")?)?;
                let text = s(ev, "text")?;
                let mut q = std::mem::take(&mut randoms);
                let mut rng = CheckedRng {
                    rng: PyRandom::new(0),
                    expected: &mut q,
                    bad: Vec::new(),
                };
                let mut next = embeds.iter();
                let mut range_bad = Vec::new();
                let remap = speaker::remap_speakers_per_line(
                    text,
                    audio_len,
                    &mut rng,
                    |start, len| {
                        let e = next
                            .next()
                            .ok_or_else(|| eyre!("more embeds than recorded"))?;
                        let want = audio_ref(e.arg("audio_tensor")?)?;
                        if want != (start, len) {
                            range_bad.push(format!(
                                "#{} want {want:?} got {:?}",
                                e.seq,
                                (start, len)
                            ));
                        }

                        let v = embedding(e.ret()?)?;
                        ensure!(v.len() == EMBED_DIM, "embedding width {}", v.len());
                        let mut out = [0f32; EMBED_DIM];
                        out.copy_from_slice(&v);
                        Ok(out)
                    },
                    f_or(ev, "threshold", 1.1)?,
                    f_or(ev, "min_duration", 1.0)?,
                )?;
                t.check(
                    "_embed_audio_segment (ranges)",
                    range_bad.is_empty(),
                    || range_bad.join("; "),
                );
                t.check(
                    "_embed_audio_segment (count)",
                    next.next().is_none(),
                    || "unused recorded embeds".into(),
                );
                let want = str_ret(ev)?;
                t.check(&ev.ev, remap.text == want, || {
                    format!("#{}: want {want:?}, got {:?}", ev.seq, remap.text)
                });
                remap_text = Some(remap.text);
            }
            _ => {}
        }
    }

    // the `transcribe()` tail end to end: join the chunk predictions, remap, merge,
    // fix, renumber, parse
    let result = set.result(stem)?;
    let chunks = result["raw_chunks"]
        .as_array()
        .ok_or_else(|| eyre!("no raw_chunks"))?;
    let preds: Vec<&str> = chunks
        .iter()
        .map(|c| c["prediction"].as_str().unwrap_or_default())
        .collect();
    let remap_in = events.iter().find(|e| e.ev == "remap_speakers_per_line");
    if let Some(r) = remap_in {
        let joined = post::join_predictions(preds.iter().copied());
        let want = s(r, "text")?;
        t.check("transcribe tail (join)", joined == want, || {
            format!("want {want:?}, got {joined:?}")
        });
    }

    let Some(remapped) = remap_text else {
        bail!("{stem}: no remap call recorded")
    };
    let out = post::finish(&preds, |_| Ok(remapped))?;
    let want_text = result["text"].as_str().unwrap_or_default();
    t.check("transcribe tail (text)", out.text == want_text, || {
        format!("want {want_text:?}, got {:?}", out.text)
    });
    let got_segs = segments_json(&out.segments);
    t.check(
        "transcribe tail (segments)",
        got_segs == result["segments"],
        || format!("want {}, got {got_segs}", result["segments"]),
    );
    Ok(())
}

fn replay_set(name: &str) -> Result<()> {
    let Some(set) = TraceSet::open(name) else {
        return Ok(());
    };
    let mut tally = Tally::default();
    let mut margins = Vec::new();
    for stem in set.stems()? {
        replay_file(&set, &stem, &mut tally, &mut margins)?;
    }

    for m in &margins {
        eprintln!("cluster margins: {m}");
    }

    let bad = tally.report(name);
    ensure!(bad == 0, "{bad} mismatches in {name}");
    Ok(())
}

#[test]
fn replay_short3() -> Result<()> {
    replay_set("short3")
}

#[test]
fn replay_e21c600() -> Result<()> {
    replay_set("e21c600")
}
