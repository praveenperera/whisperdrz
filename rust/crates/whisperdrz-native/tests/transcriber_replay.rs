//! Replays the reference traces through the Rust scheduler with a fake model that
//! serves the recorded GPU results (mel, encoder, prefill, decode-step logits, DTW
//! matrices, speaker embeddings). Every call the scheduler makes, to the model or
//! through its probe, is checked against the next recorded call of the same label:
//! decode steps (tokens, positions, slots), slot inits, prefill batches, the
//! decoder states before and after each `decode_batch`, alignment items and
//! batches, DTW inputs and outputs, window finalisation, and at the end the
//! file's `text`, `segments` and per-chunk results
//!
//! Labels are checked in order per label; the decode, DTW and finalisation
//! results feed back into the scheduler, so a divergence in one shows up in the
//! next. Each label in [`REQUIRED`] must be checked at least once per trace set

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::rc::Rc;
use std::sync::Arc;

use eyre::{Result, bail, ensure, eyre};
use half::f16;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use whisperdrz_fixtures::{ArrayData, Event as Recorded, TraceSet, as_f64};
use whisperdrz_native::decode::DecoderTables;
use whisperdrz_native::dtw::{DtwBatch, Matrices, MatricesView};
use whisperdrz_native::speaker::{EMBED_DIM, PyRandom};
use whisperdrz_native::text::Tokenizer;
use whisperdrz_native::transcriber::{
    Config, DecodeStep, DtwMatrices, Event, Logits, Mel, MelBatch, Model, ReadyMatrices, Slot,
    Transcriber, captured_buckets, get_bucket,
};

// shared with the decode tests, which use the parsing half
#[path = "support/state_json.rs"]
#[allow(dead_code)]
mod state_json;

use state_json::{same, state_fields};

/// Labels every trace set must exercise
const REQUIRED: [&str; 15] = [
    "mel",
    "_init_slot",
    "encoder",
    "prefill_flash",
    "_prefill_slots",
    "graph_forward",
    "decode_batch",
    "update_state_after_decode",
    "_prepare_slot_for_alignment",
    "_batch_align_slots",
    "fill_word_timings_batch",
    "dtw_decoder_forward",
    "_dtw_prep_batch",
    "_finalize_slot_after_alignment",
    "_embed_audio_segment",
];

/// Labels checked when the DTW is deferred: window finalisation isn't reported then,
/// since the slot has moved on by the time its text is known
fn required(defer_dtw: bool) -> impl Iterator<Item = &'static str> {
    REQUIRED
        .into_iter()
        .filter(move |&l| !(defer_dtw && l == "_finalize_slot_after_alignment"))
}

/// Where `trace_synthetic.py` sets live, relative to the reference trace root: a
/// sibling directory, since every set under the root is taken as a reference trace
const SYNTHETIC: &str = "../synth-a783291";

/// The tracer's per-label call cap (`MAX_CALLS`)
const MAX_CALLS: usize = 4000;

/// Errors kept per file before the rest are only counted
const MAX_ERRORS: usize = 20;

/// The tracer's array hash: the first 20 hex digits of sha256 over the raw bytes
fn sha20(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()[..20]
        .to_owned()
}

fn f16_bytes(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn i64_bytes(v: impl IntoIterator<Item = i64>) -> Vec<u8> {
    v.into_iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn ints(v: &Value) -> Result<Vec<i64>> {
    v.as_array()
        .ok_or_else(|| eyre!("not a list: {v}"))?
        .iter()
        .map(|x| x.as_i64().ok_or_else(|| eyre!("not an int: {x}")))
        .collect()
}

/// Numbers compare by value at any depth (python ints and floats both decode to
/// JSON numbers, and the Rust side may serialise either)
fn json_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(_), Value::Number(_)) => same(a, b),
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| json_eq(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(k, v)| y.get(k).is_some_and(|w| json_eq(v, w)))
        }
        _ => a == b,
    }
}

fn opt_f64(v: Option<f64>) -> Value {
    v.and_then(serde_json::Number::from_f64)
        .map_or(Value::Null, Value::Number)
}

fn snapshot(v: &Value) -> Result<&Map<String, Value>> {
    v.get("__state")
        .and_then(Value::as_object)
        .ok_or_else(|| eyre!("not a state snapshot"))
}

/// Differences between a slot and a python state snapshot
fn slot_diffs(slot: &Slot, snap: &Map<String, Value>) -> Vec<String> {
    let mut fields: BTreeMap<&str, Value> = state_fields(&slot.state);
    fields.insert("index", Value::from(slot.index));
    fields.insert("id", slot.id.clone().map_or(Value::Null, Value::from));
    fields.insert("last_offset", opt_f64(Some(slot.last_offset)));
    fields.insert("sofar", Value::from(slot.sofar.clone()));
    fields.insert("cutearly", Value::from(slot.cutearly));
    fields.insert("fails", Value::from(slot.fails));
    fields.insert("_continuation_count", Value::from(slot.continuation_count));
    fields.insert("is_final_chunk", Value::from(slot.is_final_chunk));

    let mut out = Vec::new();
    for (k, mine) in &fields {
        match snap.get(*k) {
            Some(py) if json_eq(py, mine) => {}
            Some(py) => out.push(format!("{k}: python {py} rust {mine}")),
            None => out.push(format!("{k}: missing in python snapshot")),
        }
    }

    let meta = snap.get("meta").and_then(Value::as_object);
    for (k, mine) in [
        ("offset", slot.meta.offset),
        ("duration", slot.meta.duration),
    ] {
        let py = meta.and_then(|m| m.get(k)).cloned().unwrap_or(Value::Null);
        if !json_eq(&py, &opt_f64(mine)) {
            out.push(format!("meta.{k}: python {py} rust {mine:?}"));
        }
    }

    out
}

/// Recorded calls of one file, consumed in order per label
struct Replay {
    set: Rc<TraceSet>,
    stem: String,
    queues: HashMap<String, VecDeque<Recorded>>,
    recorded: HashMap<String, usize>,
    /// Checks per label, across files
    checks: Rc<RefCell<BTreeMap<String, usize>>>,
    errors: Vec<String>,
    /// Encoder output row per slot (`xa_cache`), as raw fp16 bytes
    xa: HashMap<usize, Vec<u8>>,
    audio: Arc<[f32]>,
    /// The `decode_batch` event between its before and after probes
    decode: Option<Recorded>,
    buckets: Vec<usize>,
    /// Crops the planner prefetched for the current file, and those since embedded
    prefetched: HashSet<(usize, usize)>,
    embedded: HashSet<(usize, usize)>,
    /// Crops the remap embedded, over every file
    crop_embeds: usize,
    /// Hand DTW matrices back as still in flight, so the scheduler takes its deferred
    /// path: window state moves on at once and the texts follow from a helper thread
    defer_dtw: bool,
}

impl Replay {
    /// [`slot_diffs`], except that a deferred alignment appends a continuing window's
    /// text at the next decode step, after the prefill and decode events in between
    /// report the slot, so `sofar` is checked through the results instead
    fn slot_diffs(&self, slot: &Slot, snap: &Map<String, Value>) -> Vec<String> {
        let mut diffs = slot_diffs(slot, snap);
        if self.defer_dtw {
            diffs.retain(|d| !d.starts_with("sofar: "));
        }
        diffs
    }

    fn check(&mut self, label: &str, ok: bool, msg: impl FnOnce() -> String) {
        *self
            .checks
            .borrow_mut()
            .entry(label.to_owned())
            .or_default() += 1;
        if !ok {
            if self.errors.len() < MAX_ERRORS {
                self.errors
                    .push(format!("{} {label}: {}", self.stem, msg()));
            } else {
                self.errors.push(String::new());
            }
        }
    }

    /// The next recorded call of `label`; `None` past the tracer's cap
    fn pop(&mut self, label: &str) -> Result<Option<Recorded>> {
        if let Some(ev) = self.queues.get_mut(label).and_then(VecDeque::pop_front) {
            return Ok(Some(ev));
        }

        if self.recorded.get(label).copied().unwrap_or(0) >= MAX_CALLS {
            return Ok(None);
        }

        bail!(
            "{}: the scheduler made more {label} calls than recorded",
            self.stem
        )
    }

    fn peek(&self, label: &str) -> Result<&Recorded> {
        self.queues
            .get(label)
            .and_then(VecDeque::front)
            .ok_or_else(|| eyre!("{}: no recorded {label} left", self.stem))
    }

    fn mel(&mut self, audio: &[f32]) -> Result<Mel> {
        let ev = self.pop("mel")?.ok_or_else(|| eyre!("mel past the cap"))?;
        let wave = ev.arg("waveform")?;
        let sha = wave["blob"].as_str().unwrap_or_default().to_owned();
        let shape = ints(&wave["shape"])?;
        let ok = shape == [1, audio.len() as i64] && sha20(&f32_bytes(audio)) == sha;
        self.check("mel", ok, || format!("#{} waveform differs", ev.seq));

        let arr = self.set.array(ev.ret()?)?;
        ensure!(
            arr.shape.len() == 3 && arr.shape[0] == 1,
            "mel shape {:?}",
            arr.shape
        );
        Mel::new(arr.shape[1], arr.shape[2], arr.to_f32()?)
    }

    fn on_event(&mut self, event: &Event<'_>) -> Result<()> {
        match event {
            Event::InitSlot { slot, chunk } => {
                let Some(ev) = self.pop("_init_slot")? else {
                    return Ok(());
                };
                let c = ev.arg("chunk")?;
                let range = ints(&c["audio"]["__audio"])?;
                let ok = ev.arg("slot_idx")?.as_u64() == Some(slot.index as u64)
                    && c["id"].as_str() == Some(chunk.id.as_str())
                    && as_f64(&c["duration"]) == chunk.duration
                    && as_f64(&c["offset"]) == chunk.offset
                    && range.get(1).copied() == Some(chunk.audio.len() as i64)
                    && chunk.offset.map(|o| (o * 16000.0).round() as i64) == range.first().copied();
                self.check("_init_slot", ok, || {
                    format!(
                        "#{}: python slot {} {c}, rust slot {} {}",
                        ev.seq,
                        ev.arg("slot_idx").unwrap(),
                        slot.index,
                        chunk.id
                    )
                });
            }
            Event::PrefillSlots { slots } => {
                let Some(ev) = self.pop("_prefill_slots")? else {
                    return Ok(());
                };
                let want = ints(ev.arg("slots")?)?;
                let got: Vec<i64> = slots.iter().map(|&s| s as i64).collect();
                self.check("_prefill_slots", want == got, || {
                    format!("#{}: python {want:?} rust {got:?}", ev.seq)
                });
            }
            Event::DecodeBefore { slots, all, tokens } => {
                let ev = self
                    .pop("decode_batch")?
                    .ok_or_else(|| eyre!("decode_batch past the cap"))?;
                let want = ints(ev.arg("slot_indices")?)?;
                let got: Vec<i64> = slots.iter().map(|&s| s as i64).collect();
                self.check("decode_batch", want == got, || {
                    format!("#{}: slots python {want:?} rust {got:?}", ev.seq)
                });

                let states = ev
                    .arg("states")?
                    .as_array()
                    .ok_or_else(|| eyre!("states"))?;
                let counts = ints(ev.arg("token_counts")?)?;
                let last = ints(ev.arg("last_token_ids")?)?;
                let mut diffs = Vec::new();
                for (r, &s) in slots.iter().enumerate() {
                    let slot = &all[s];
                    if let Some(snap) = states.get(r) {
                        diffs.extend(
                            self.slot_diffs(slot, snapshot(snap)?)
                                .into_iter()
                                .map(|d| format!("row {r}: {d}")),
                        );
                    }

                    if counts.get(r) != Some(&(slot.state.token_count as i64))
                        || last.get(r) != Some(&(slot.state.last_token_id as i64))
                    {
                        diffs.push(format!("row {r}: token_counts/last_token_ids"));
                    }
                }

                // the repetition check's all_tokens windows
                let rows = ev.arg("all_tokens_gpu")?["__rows"]
                    .as_object()
                    .ok_or_else(|| eyre!("no __rows"))?;
                for (k, w) in rows {
                    let s: usize = k.parse()?;
                    let n = all[s].state.token_count;
                    let mine: Vec<i64> = tokens[s][n.saturating_sub(100)..n]
                        .iter()
                        .map(|&t| t as i64)
                        .collect();
                    if ints(w)? != mine {
                        diffs.push(format!("all_tokens window of slot {s}"));
                    }
                }

                self.check("decode_batch", diffs.is_empty(), || {
                    format!("#{} before: {diffs:?}", ev.seq)
                });
                self.decode = Some(ev);
            }
            Event::DecodeAfter {
                slots,
                decisions,
                all,
            } => {
                let ev = self
                    .decode
                    .take()
                    .ok_or_else(|| eyre!("decode after without before"))?;
                let ret = ev.ret()?.as_array().ok_or_else(|| eyre!("decode ret"))?;
                let after = ev
                    .after_arg("states")?
                    .as_array()
                    .ok_or_else(|| eyre!("after states"))?;
                let mut diffs = Vec::new();
                for (r, (&s, d)) in slots.iter().zip(decisions.iter()).enumerate() {
                    let tup = ret.get(r).and_then(|t| t["__tuple"].as_array());
                    let ok = tup.is_some_and(|t| {
                        t.first().and_then(Value::as_str) == Some(d.text.as_str())
                            && t.get(1).and_then(Value::as_u64) == Some(d.token as u64)
                    });
                    if !ok {
                        diffs.push(format!(
                            "row {r}: python {:?} rust ({:?}, {})",
                            ret.get(r),
                            d.text,
                            d.token
                        ));
                    }

                    if let Some(snap) = after.get(r) {
                        diffs.extend(
                            self.slot_diffs(&all[s], snapshot(snap)?)
                                .into_iter()
                                .map(|d| format!("row {r} after: {d}")),
                        );
                    }
                }

                self.check("decode_batch", diffs.is_empty(), || {
                    format!("#{}: {diffs:?}", ev.seq)
                });
            }
            Event::UpdateState { slot, text, token } => {
                let Some(ev) = self.pop("update_state_after_decode")? else {
                    return Ok(());
                };
                let mut diffs = Vec::new();
                if ev.arg("newtext")?.as_str() != Some(text)
                    || ev.arg("newtoken_id")?.as_u64() != Some(*token as u64)
                {
                    diffs.push(format!(
                        "args python {:?} {:?} rust {text:?} {token}",
                        ev.arg("newtext")?,
                        ev.arg("newtoken_id")?
                    ));
                }

                diffs.extend(self.slot_diffs(slot, snapshot(ev.after_arg("state")?)?));
                self.check("update_state_after_decode", diffs.is_empty(), || {
                    format!("#{}: {diffs:?}", ev.seq)
                });
            }
            Event::PrepareAlignment { slot, item } => {
                let Some(ev) = self.pop("_prepare_slot_for_alignment")? else {
                    return Ok(());
                };
                let ret = ev.ret()?;
                let mut diffs = self.slot_diffs(slot, snapshot(ev.after_arg("s")?)?);
                match item {
                    None => {
                        if !ret.is_null() {
                            diffs.push("python returned an item".into());
                        }
                    }
                    Some(item) => {
                        let want = [
                            ("cur", Value::from(item.cur.clone())),
                            ("clip_to_last_timing", Value::from(item.clip_to_last_timing)),
                            ("more", Value::from(item.more)),
                            ("phase", Value::from(item.phase.as_str())),
                        ];
                        for (k, v) in want {
                            if !ret.get(k).is_some_and(|p| json_eq(p, &v)) {
                                diffs.push(format!("{k}: python {:?} rust {v}", ret.get(k)));
                            }
                        }

                        let py = ret
                            .get("last_timing_override")
                            .cloned()
                            .unwrap_or(Value::Null);
                        if !json_eq(&py, &opt_f64(item.last_timing_override)) {
                            diffs.push(format!(
                                "last_timing_override: python {py} rust {:?}",
                                item.last_timing_override
                            ));
                        }
                    }
                }

                self.check("_prepare_slot_for_alignment", diffs.is_empty(), || {
                    format!("#{}: {diffs:?}", ev.seq)
                });
            }
            Event::FillWordTimings {
                xa_slots,
                texts,
                out,
            } => {
                let Some(ev) = self.pop("fill_word_timings_batch")? else {
                    return Ok(());
                };
                let data = ev
                    .arg("slot_data")?
                    .as_array()
                    .ok_or_else(|| eyre!("slot_data"))?;
                let mut diffs = Vec::new();
                if data.len() != texts.len() {
                    diffs.push(format!("{} slots, rust {}", data.len(), texts.len()));
                }

                for (i, d) in data.iter().enumerate() {
                    let tup = d["__tuple"]
                        .as_array()
                        .ok_or_else(|| eyre!("slot_data tuple"))?;
                    let Some((text, frames)) = texts.get(i) else {
                        break;
                    };
                    if tup[1].as_str() != Some(text.as_str()) || tup[2].as_i64() != Some(*frames) {
                        diffs.push(format!(
                            "slot {i}: python ({:?}, {}) rust ({text:?}, {frames})",
                            tup[1], tup[2]
                        ));
                    }

                    // deferred, the event comes once the texts land, by when a later
                    // window may have replaced the slot's encoder output
                    if self.defer_dtw {
                        continue;
                    }

                    let sha = tup[0]["blob"].as_str().unwrap_or_default();
                    let mine = self
                        .xa
                        .get(&xa_slots[i])
                        .map(|x| sha20(x))
                        .unwrap_or_default();
                    if sha != mine {
                        diffs.push(format!("slot {i}: xa of slot {} differs", xa_slots[i]));
                    }
                }

                let ret = ev.ret()?;
                let got = Value::from(out.to_vec());
                if !json_eq(ret, &got) {
                    diffs.push(format!("ret python {ret} rust {got}"));
                }

                self.check("fill_word_timings_batch", diffs.is_empty(), || {
                    format!("#{}: {diffs:?}", ev.seq)
                });
            }
            Event::Finalize {
                slot,
                out,
                last_timing,
                more,
                phase,
                pre_augmented,
                result,
            } => {
                let Some(ev) = self.pop("_finalize_slot_after_alignment")? else {
                    return Ok(());
                };
                let mut diffs = self.slot_diffs(slot, snapshot(ev.after_arg("s")?)?);
                let args = [
                    ("out", Value::from(*out)),
                    ("last_timing", opt_f64(Some(*last_timing))),
                    ("more", Value::from(*more)),
                    ("phase", Value::from(*phase)),
                    (
                        "pre_augmented_cur",
                        pre_augmented.map_or(Value::Null, Value::from),
                    ),
                ];
                for (k, v) in args {
                    if !json_eq(ev.arg(k)?, &v) {
                        diffs.push(format!("{k}: python {} rust {v}", ev.arg(k)?));
                    }
                }

                diffs.extend(result_diffs(ev.ret()?, *result));
                self.check("_finalize_slot_after_alignment", diffs.is_empty(), || {
                    format!("#{}: {diffs:?}", ev.seq)
                });
            }
            Event::BatchAlign { slots, results } => {
                let Some(ev) = self.pop("_batch_align_slots")? else {
                    return Ok(());
                };
                let items = ev
                    .arg("pending_align")?
                    .as_array()
                    .ok_or_else(|| eyre!("pending_align"))?;
                let want: Vec<Option<u64>> = items
                    .iter()
                    .map(|i| snapshot(&i["state"]).ok().and_then(|s| s["index"].as_u64()))
                    .collect();
                let got: Vec<Option<u64>> = slots.iter().map(|&s| Some(s as u64)).collect();
                let mut diffs = Vec::new();
                if want != got {
                    diffs.push(format!("slots python {want:?} rust {got:?}"));
                }

                let ret = ev.ret()?.as_array().ok_or_else(|| eyre!("ret"))?;
                if ret.len() != results.len() {
                    diffs.push(format!("{} results, rust {}", ret.len(), results.len()));
                }

                for (py, mine) in ret.iter().zip(results.iter()) {
                    diffs.extend(result_diffs(py, mine.as_ref()));
                }

                self.check("_batch_align_slots", diffs.is_empty(), || {
                    format!("#{}: {diffs:?}", ev.seq)
                });
            }
        }

        Ok(())
    }
}

/// Differences between a python result dict (or `None`) and a Rust result
fn result_diffs(
    py: &Value,
    mine: Option<&whisperdrz_native::transcriber::ChunkResult>,
) -> Vec<String> {
    let Some(r) = mine else {
        return if py.is_null() {
            Vec::new()
        } else {
            vec![format!("python returned {py}")]
        };
    };
    if py.is_null() {
        return vec![format!("python returned None, rust {}", r.id)];
    }

    let mut out = Vec::new();
    let fields = [
        ("id", Value::from(r.id.clone())),
        ("prediction", Value::from(r.prediction.clone())),
        ("transcription", Value::from(r.prediction.clone())),
        ("stop_reason", Value::from(r.stop_reason.clone())),
        ("language", Value::from(r.language.clone())),
        (
            "raw_cur",
            r.raw_cur.clone().map_or(Value::Null, Value::from),
        ),
    ];
    for (k, v) in fields {
        if !py.get(k).is_some_and(|p| json_eq(p, &v)) {
            out.push(format!("result {k}: python {:?} rust {v}", py.get(k)));
        }
    }

    if let Some(map) = &r.speaker_id_map {
        let want: Vec<Value> = map
            .iter()
            .enumerate()
            .map(|(i, &orig)| Value::from(vec![Value::from(i), Value::from(orig)]))
            .collect();
        let py_map = py["speaker_id_map"]
            .get("__items")
            .cloned()
            .unwrap_or_else(|| Value::Array(Vec::new()));
        if !json_eq(&py_map, &Value::from(want.clone())) {
            out.push(format!("speaker_id_map: python {py_map} rust {want:?}"));
        }
    }

    out
}

/// Serves the recorded GPU results in call order
struct FakeModel(Rc<RefCell<Replay>>);

/// Recorded matrices that claim to still be on their way
struct InFlight(Matrices);

/// How long the fake DTW thread waits for its matrices: long against a replayed decode
/// step, so the texts land several steps after their launch, as on a GPU whose forward
/// runs beside the steps, and the scheduler's joins on demand are what keeps them exact
const IN_FLIGHT: std::time::Duration = std::time::Duration::from_millis(3);

impl DtwMatrices for InFlight {
    fn ready(&self) -> bool {
        false
    }

    fn wait(&mut self) -> Result<MatricesView<'_>> {
        std::thread::sleep(IN_FLIGHT);
        Ok(self.0.view())
    }
}

impl Model for FakeModel {
    fn reset_kv(&mut self, _slots: &[usize]) -> Result<()> {
        // `reset_flash_kv` isn't traced; its effect shows in the next positions
        Ok(())
    }

    fn encode(&mut self, mel: &MelBatch, slots: &[usize]) -> Result<()> {
        let mut r = self.0.borrow_mut();
        let ev = r
            .pop("encoder")?
            .ok_or_else(|| eyre!("encoder past the cap"))?;
        let x = ev.arg("x")?;
        let ok = ints(&x["shape"])? == [mel.rows as i64, mel.n_mels as i64, 3000]
            && x["blob"].as_str() == Some(sha20(&f16_bytes(&mel.data)).as_str());
        r.check("encoder", ok, || {
            format!("#{}: mel batch for slots {slots:?} differs", ev.seq)
        });
        ensure!(ok, "{}: encoder input differs at #{}", r.stem, ev.seq);

        let out = r.set.array(ev.ret()?)?;
        let ArrayData::F16(data) = out.data else {
            bail!("encoder output is {}", out.dtype)
        };
        ensure!(out.shape[0] == slots.len(), "encoder rows");
        let per = data.len() / slots.len();
        for (i, &s) in slots.iter().enumerate() {
            r.xa.insert(s, f16_bytes(&data[i * per..(i + 1) * per]));
        }

        Ok(())
    }

    fn prefill(&mut self, tokens: &[u32], slots: &[usize]) -> Result<Logits> {
        let mut r = self.0.borrow_mut();
        let ev = r
            .pop("prefill_flash")?
            .ok_or_else(|| eyre!("prefill past the cap"))?;
        let want_tokens = r.set.array(ev.arg("tokens")?)?.to_i64()?;
        let want_slots = r.set.array(ev.arg("slot_indices")?)?.to_i64()?;
        let got_tokens: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
        let got_slots: Vec<i64> = slots.iter().map(|&s| s as i64).collect();
        // prefill attends to the encoder output it was just given
        let xa_sha = ev.arg("xa")?["blob"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        let mine: Vec<u8> = slots.iter().flat_map(|s| r.xa[s].clone()).collect();
        let ok = want_tokens == got_tokens && want_slots == got_slots && sha20(&mine) == xa_sha;
        r.check("prefill_flash", ok, || {
            format!(
                "#{}: python {want_tokens:?} {want_slots:?}, rust {got_tokens:?} {got_slots:?}",
                ev.seq
            )
        });
        ensure!(ok, "{}: prefill differs at #{}", r.stem, ev.seq);

        let out = r.set.array(ev.ret()?)?;
        let ArrayData::F16(data) = out.data else {
            bail!("prefill logits are {}", out.dtype)
        };
        let [n, t, v] = out.shape[..] else {
            bail!("prefill logits shape {:?}", out.shape)
        };
        let mut last = Vec::with_capacity(n * v);
        for row in 0..n {
            let base = (row * t + t - 1) * v;
            last.extend_from_slice(&data[base..base + v]);
        }

        Logits::new(n, v, last)
    }

    fn decode_step(&mut self, step: &DecodeStep) -> Result<Logits> {
        let mut r = self.0.borrow_mut();
        let ev = r
            .pop("graph_forward")?
            .ok_or_else(|| eyre!("graph_forward past the cap"))?;
        let n = step.rows;
        let want_tokens = r.set.array(ev.arg("tokens")?)?.to_i64()?;
        let want_pos = r.set.array(ev.arg("positions")?)?.to_i64()?;
        let want_slots = r.set.array(ev.arg("slot_indices")?)?.to_i64()?;
        let got_tokens: Vec<i64> = step.tokens[..n].iter().map(|&t| t as i64).collect();
        let got_pos: Vec<i64> = step.positions[..n].iter().map(|&p| p as i64).collect();
        let got_slots: Vec<i64> = step.slots[..n].iter().map(|&s| s as i64).collect();
        // padding: the bench's bucket for this size, rows past n zeroed onto the trash slot
        let bucket = get_bucket(n, &r.buckets);
        let padded = step.bucket == bucket
            && step.tokens.len() == bucket
            && step.tokens[n..].iter().all(|&t| t == 0)
            && step.positions[n..].iter().all(|&p| p == 0)
            && step.slots[n..].iter().all(|&s| s == step.trash_slot);
        let ok =
            want_tokens == got_tokens && want_pos == got_pos && want_slots == got_slots && padded;
        r.check("graph_forward", ok, || {
            format!(
                "#{}: python {want_tokens:?} {want_pos:?} {want_slots:?}, rust {got_tokens:?} {got_pos:?} {got_slots:?} bucket {}",
                ev.seq, step.bucket
            )
        });
        ensure!(ok, "{}: decode step differs at #{}", r.stem, ev.seq);

        // the logits are recorded as decode_batch's input, the call that follows
        let db = r.peek("decode_batch")?;
        ensure!(
            db.depth == 0,
            "{}: decode_batch #{} is not a decode step's",
            r.stem,
            db.seq
        );
        let arr = r.set.array(db.arg("logits")?)?;
        let ArrayData::F16(data) = arr.data else {
            bail!("logits are {}", arr.dtype)
        };
        let v = *arr.shape.last().ok_or_else(|| eyre!("scalar logits"))?;
        Logits::new(arr.shape[0], v, data)
    }

    fn dtw_prep(&mut self, batch: &DtwBatch, xa_slots: &[usize]) -> Result<Matrices> {
        let mut r = self.0.borrow_mut();
        let fwd = r
            .pop("dtw_decoder_forward")?
            .ok_or_else(|| eyre!("dtw forward past the cap"))?;
        let x = fwd.arg("x")?;
        let toks = batch.full_tokens.iter().map(|&t| t as i64);
        let tokens_ok = ints(&x["shape"])? == [batch.rows() as i64, batch.t_pad as i64]
            && match x.get("data") {
                Some(_) => r.set.array(x)?.to_i64()? == toks.collect::<Vec<_>>(),
                None => x["blob"].as_str() == Some(sha20(&i64_bytes(toks)).as_str()),
            };
        let xa: Vec<u8> = xa_slots.iter().flat_map(|s| r.xa[s].clone()).collect();
        let xa_ok = fwd.arg("xa")?["blob"].as_str() == Some(sha20(&xa).as_str());
        r.check("dtw_decoder_forward", tokens_ok && xa_ok, || {
            format!(
                "#{}: tokens ok {tokens_ok}, xa ok {xa_ok} (slots {xa_slots:?})",
                fwd.seq
            )
        });

        let prep = r
            .pop("_dtw_prep_batch")?
            .ok_or_else(|| eyre!("dtw prep past the cap"))?;
        let lens: Vec<i64> = batch.actual_lens.iter().map(|&l| l as i64).collect();
        let caps: Vec<i64> = batch.audio_caps.iter().map(|&c| c as i64).collect();
        let ok = ints(prep.arg("actual_lens")?)? == lens && ints(prep.arg("audio_caps")?)? == caps;
        r.check("_dtw_prep_batch", ok, || {
            format!(
                "#{}: lens/caps python {:?} {:?} rust {lens:?} {caps:?}",
                prep.seq,
                prep.args.get("actual_lens"),
                prep.args.get("audio_caps")
            )
        });
        ensure!(
            tokens_ok && xa_ok && ok,
            "{}: DTW batch differs at #{}",
            r.stem,
            fwd.seq
        );

        let m = r.set.array(prep.ret()?)?;
        let [b, t, a] = m.shape[..] else {
            bail!("prep shape {:?}", m.shape)
        };
        Ok(Matrices {
            shape: [b, t, a],
            data: m.to_f32()?,
        })
    }

    fn dtw_launch(&mut self, batch: &DtwBatch, xa_slots: &[usize]) -> Result<Box<dyn DtwMatrices>> {
        let m = self.dtw_prep(batch, xa_slots)?;
        Ok(if self.0.borrow().defer_dtw {
            Box::new(InFlight(m))
        } else {
            Box::new(ReadyMatrices(m))
        })
    }

    fn reset_embeds(&mut self) -> Result<()> {
        let mut r = self.0.borrow_mut();
        // every prefetched crop must be one the remap embeds, so early work is never wasted
        let unused: Vec<_> = r.prefetched.difference(&r.embedded).copied().collect();
        r.check("prefetch_unused", unused.is_empty(), || {
            format!("prefetched but never embedded: {unused:?}")
        });
        r.prefetched.clear();
        r.embedded.clear();
        Ok(())
    }

    fn prefetch_embed(&mut self, audio: &Arc<[f32]>, start: usize, len: usize) -> Result<()> {
        let mut r = self.0.borrow_mut();
        ensure!(
            Arc::ptr_eq(audio, &r.audio) || **audio == *r.audio,
            "prefetch from other audio"
        );
        r.prefetched.insert((start, len));
        Ok(())
    }

    fn embed_crop(
        &mut self,
        audio: &Arc<[f32]>,
        start: usize,
        len: usize,
    ) -> Result<[f32; EMBED_DIM]> {
        {
            let mut r = self.0.borrow_mut();
            // the planner must have named every crop before the remap asks for it
            let hit = r.prefetched.contains(&(start, len));
            r.check("prefetch_hit", hit, || {
                format!("crop {start}+{len} was never prefetched")
            });
            r.embedded.insert((start, len));
            r.crop_embeds += 1;
        }

        self.embed(&audio[start..start + len])
    }

    fn embed(&mut self, audio: &[f32]) -> Result<[f32; EMBED_DIM]> {
        let mut r = self.0.borrow_mut();
        let ev = r
            .pop("_embed_audio_segment")?
            .ok_or_else(|| eyre!("embed past the cap"))?;
        let range = ints(&ev.arg("audio_tensor")?["__audio"])?;
        let (start, len) = (range[0] as usize, range[1] as usize);
        let ok = len == audio.len() && r.audio.get(start..start + len) == Some(audio);
        r.check("_embed_audio_segment", ok, || {
            format!(
                "#{}: python range {start}+{len}, rust {} samples",
                ev.seq,
                audio.len()
            )
        });

        let v = r.set.array(ev.ret()?)?.to_f32()?;
        v.try_into()
            .map_err(|v: Vec<f32>| eyre!("embedding width {}", v.len()))
    }
}

/// The file's audio, rebuilt from the chunk waveforms the `mel` calls recorded
fn file_audio(set: &TraceSet, events: &[Recorded], samples: usize) -> Result<Vec<f32>> {
    let mels: Vec<&Recorded> = events.iter().filter(|e| e.ev == "mel").collect();
    let inits: Vec<&Recorded> = events.iter().filter(|e| e.ev == "_init_slot").collect();
    ensure!(
        mels.len() == inits.len(),
        "{} mel calls for {} inits",
        mels.len(),
        inits.len()
    );

    let mut pieces = Vec::new();
    for (mel, init) in mels.iter().zip(&inits) {
        let range = ints(&init.arg("chunk")?["audio"]["__audio"])?;
        let wave = set.array(mel.arg("waveform")?)?.to_f32()?;
        ensure!(wave.len() as i64 == range[1], "mel waveform length");
        pieces.push((range[0] as usize, wave));
    }

    pieces.sort_by_key(|(start, _)| *start);
    let mut audio = Vec::with_capacity(samples);
    for (start, wave) in pieces {
        ensure!(
            start == audio.len(),
            "chunk at {start} after {} samples",
            audio.len()
        );
        audio.extend(wave);
    }

    ensure!(
        audio.len() == samples,
        "rebuilt {} samples, want {samples}",
        audio.len()
    );
    Ok(audio)
}

fn replay_file(
    set: &Rc<TraceSet>,
    stem: &str,
    config: &Config,
    tables: &Arc<DecoderTables>,
    checks: &Rc<RefCell<BTreeMap<String, usize>>>,
    defer_dtw: bool,
) -> Result<()> {
    replay_recorded(set, stem, config, tables, checks, defer_dtw, None)
}

/// [`replay_file`], also filling `oracle` with every result the replay served, keyed by
/// content, when given
#[allow(clippy::too_many_arguments)]
fn replay_recorded(
    set: &Rc<TraceSet>,
    stem: &str,
    config: &Config,
    tables: &Arc<DecoderTables>,
    checks: &Rc<RefCell<BTreeMap<String, usize>>>,
    defer_dtw: bool,
    oracle: Option<Rc<RefCell<Oracle>>>,
) -> Result<()> {
    let events = set.events(stem)?;
    let result = set.result(stem)?;
    let samples = result["samples"]
        .as_u64()
        .ok_or_else(|| eyre!("no samples"))? as usize;
    let audio: Arc<[f32]> = file_audio(set, &events, samples)?.into();

    let mut queues: HashMap<String, VecDeque<Recorded>> = HashMap::new();
    for ev in events {
        queues.entry(ev.ev.clone()).or_default().push_back(ev);
    }

    let recorded = queues.iter().map(|(k, v)| (k.clone(), v.len())).collect();
    let config = config.clone();
    let replay = Rc::new(RefCell::new(Replay {
        set: set.clone(),
        stem: stem.to_owned(),
        queues,
        recorded,
        checks: checks.clone(),
        errors: Vec::new(),
        xa: HashMap::new(),
        audio: audio.clone(),
        decode: None,
        buckets: captured_buckets(config.bs),
        prefetched: HashSet::new(),
        embedded: HashSet::new(),
        crop_embeds: 0,
        defer_dtw,
    }));

    let mel_replay = replay.clone();
    let mel_oracle = oracle.clone();
    let mut trn = Transcriber::new(
        config,
        Recorder::new(FakeModel(replay.clone()), oracle.clone()),
        Box::new(move |a: &[f32]| {
            let mel = mel_replay.borrow_mut().mel(a)?;
            if let Some(o) = &mel_oracle {
                o.borrow_mut().mel.insert(sha20(&f32_bytes(a)), mel.clone());
            }
            Ok(mel)
        }),
        tables.clone(),
    )?;
    let probe_replay = replay.clone();
    let probe_error: Rc<RefCell<Option<eyre::Report>>> = Rc::default();
    let probe_err = probe_error.clone();
    trn.set_probe(Box::new(move |event| {
        if let (Some(o), Event::UpdateState { slot, text, token }) = (&oracle, event) {
            o.borrow_mut().tokens.record(slot, text, *token);
        }
        if let Err(e) = probe_replay.borrow_mut().on_event(event) {
            probe_err.borrow_mut().get_or_insert(e);
        }
    }));

    // the reference seeds the global `random` before each file
    let mut rng = PyRandom::new(0);
    let out = trn.transcribe(&audio, &mut rng);
    if let Some(e) = probe_error.borrow_mut().take() {
        return Err(e.wrap_err(format!("{stem}: probe")));
    }

    let r = replay.borrow();
    if !r.errors.is_empty() {
        let shown: Vec<&String> = r.errors.iter().filter(|e| !e.is_empty()).collect();
        bail!(
            "{stem}: {} mismatches:\n{}",
            r.errors.len(),
            shown
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    let out = out?;
    for label in required(defer_dtw) {
        let left = r.queues.get(label).map_or(0, VecDeque::len);
        ensure!(
            left == 0,
            "{stem}: {left} recorded {label} calls never made"
        );
    }

    ensure!(
        out.text == result["text"].as_str().unwrap_or_default(),
        "{stem}: text differs\npython {:?}\nrust   {:?}",
        result["text"],
        out.text
    );
    let segments = serde_json::to_value(&out.segments)?;
    ensure!(
        json_eq(&segments, &result["segments"]),
        "{stem}: segments differ"
    );

    let raw = result["raw_chunks"]
        .as_array()
        .ok_or_else(|| eyre!("raw_chunks"))?;
    ensure!(raw.len() == out.raw_chunks.len(), "{stem}: chunk count");
    for (py, mine) in raw.iter().zip(&out.raw_chunks) {
        let diffs = result_diffs(py, Some(mine));
        ensure!(
            diffs.is_empty(),
            "{stem}: raw_chunks {}: {diffs:?}",
            mine.id
        );
    }

    eprintln!(
        "{stem}: text and {} segments match, {} chunks",
        out.segments.len(),
        out.raw_chunks.len()
    );
    Ok(())
}

fn replay(name: &str, config: Config) -> Result<()> {
    replay_with(name, config, false)
}

fn replay_with(name: &str, config: Config, defer_dtw: bool) -> Result<()> {
    let Some(set) = TraceSet::open(name) else {
        return Ok(());
    };
    let set = Rc::new(set);
    let tables = Arc::new(DecoderTables::new(Tokenizer::multilingual()?)?);
    let checks: Rc<RefCell<BTreeMap<String, usize>>> = Rc::default();
    for stem in set.stems()? {
        replay_file(&set, &stem, &config, &tables, &checks, defer_dtw)?;
    }

    let checks = checks.borrow();
    eprintln!("{name} checks per label: {checks:?}");
    for label in required(defer_dtw) {
        ensure!(
            checks.get(label).copied().unwrap_or(0) > 0,
            "{name}: {label} never checked"
        );
    }

    // every remap crop went through `embed_crop` after a prefetch named it
    ensure!(
        checks.get("prefetch_hit").copied().unwrap_or(0) > 0,
        "{name}: no remap crop was checked against the prefetches"
    );

    Ok(())
}

#[test]
fn replay_short3() -> Result<()> {
    replay("short3", Config::default())
}

#[test]
fn replay_e21c600() -> Result<()> {
    replay("e21c600", Config::default())
}

/// The deferred alignment path (a GPU model's): each window's state moves on before
/// its DTW lands, and the texts and results, checked against the reference's, follow
/// from a helper thread some decode steps later
#[test]
fn replay_short3_deferred_dtw() -> Result<()> {
    replay_with("short3", Config::default(), true)
}

#[test]
fn replay_e21c600_deferred_dtw() -> Result<()> {
    replay_with("e21c600", Config::default(), true)
}

/// `rust/tools/trace_synthetic.py`: the Python scheduler on the CPU with BS=4 and a
/// stand-in model, so freed slots are refilled in CPython set order and windows
/// continue often
#[test]
fn replay_synthetic_bs4() -> Result<()> {
    replay(
        &format!("{SYNTHETIC}/synth-bs4"),
        Config {
            bs: 4,
            ..Config::default()
        },
    )
}

/// The deferred alignment path where windows continue often and freed slots refill
/// while a finished chunk's text is still on the DTW thread
#[test]
fn replay_synthetic_bs4_deferred_dtw() -> Result<()> {
    replay_with(
        &format!("{SYNTHETIC}/synth-bs4"),
        Config {
            bs: 4,
            ..Config::default()
        },
        true,
    )
}

/// The same with BS=6 and identical logits rows per call: slots that start together
/// free up together, so refills pick several slots from the set at once, in CPython's
/// table order rather than ascending
#[test]
fn replay_synthetic_bs6_lockstep() -> Result<()> {
    replay(
        &format!("{SYNTHETIC}/synth-bs6-lockstep"),
        Config {
            bs: 6,
            ..Config::default()
        },
    )
}

// ---------------------------------------------------------------------------------------
// Speculative decoding proof
//
// The reference replay serves results in call order, which only a scheduler making the
// reference's calls can follow. With speculation on, steps carry several tokens a row and
// windows end at other steps, so batches differ. The proof records, during a passing
// reference replay, every result keyed by content instead: the encoder input row of each
// slot's window, the token prefix fed at each position, the DTW row's tokens, the audio of
// each crop. A speculative run then gets the answer an ideal model would give for each
// position from its own prefix, and has to emit every token and output of the reference
// run. Logits of a prefix the reference never ran (a rejected draft's position) exist
// only as an error, raised if a decision reads them
// ---------------------------------------------------------------------------------------

use whisperdrz_native::transcriber::{SpecConfig, SpecStats, StepLogits, VerifyStep};

/// A slot's window, by its encoder input row
type WindowKey = String;

/// The tokens each window emitted, in order, by chunk id and window offset
#[derive(Debug, Default, Clone, PartialEq)]
struct WindowTokens(BTreeMap<(String, u64), Vec<(u32, String)>>);

impl WindowTokens {
    fn record(&mut self, slot: &Slot, text: &str, token: u32) {
        let key = (
            slot.id.clone().unwrap_or_default(),
            slot.state.offset.to_bits(),
        );
        self.0
            .entry(key)
            .or_default()
            .push((token, text.to_owned()));
    }

    /// Fails on the first window or token that differs
    fn check_same(&self, want: &Self, stem: &str) -> Result<usize> {
        let got_keys: Vec<_> = self.0.keys().collect();
        let want_keys: Vec<_> = want.0.keys().collect();
        ensure!(
            got_keys == want_keys,
            "{stem}: windows differ\nreference {want_keys:?}\nspec      {got_keys:?}"
        );
        let mut n = 0;
        for (key, w) in &want.0 {
            let g = &self.0[key];
            ensure!(
                g.len() == w.len(),
                "{stem}: window {key:?} emitted {} tokens, the reference {}",
                g.len(),
                w.len()
            );
            for (i, (a, b)) in g.iter().zip(w).enumerate() {
                ensure!(
                    a == b,
                    "{stem}: window {key:?} token {i}: spec {a:?}, reference {b:?}"
                );
            }
            n += w.len();
        }
        Ok(n)
    }
}

/// Every result a reference replay served, keyed by content
#[derive(Default)]
struct Oracle {
    /// Waveform hash to mel
    mel: HashMap<String, Mel>,
    /// Encoder input rows seen
    windows: HashSet<WindowKey>,
    /// (window, start tokens) to the prefill's last-position logits
    prefill: HashMap<(WindowKey, Vec<u32>), Vec<f16>>,
    /// (window, token prefix hash) to the logits after that prefix
    logits: HashMap<(WindowKey, u64), Vec<f16>>,
    /// (window, the DTW row's real tokens, audio cap) to its `[tokens, cap]` matrix rows
    dtw: HashMap<(WindowKey, Vec<u32>, usize), Vec<f32>>,
    /// Audio hash to embedding
    embed: HashMap<String, [f32; EMBED_DIM]>,
    /// The tokens the reference run emitted
    tokens: WindowTokens,
}

/// Each slot's window and the tokens its KV holds, as a model sees them
#[derive(Default)]
struct SlotTrack {
    window: HashMap<usize, WindowKey>,
    tokens: HashMap<usize, Vec<u32>>,
}

fn prefix_hash(tokens: &[u32]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    tokens.hash(&mut h);
    h.finish()
}

impl SlotTrack {
    fn reset(&mut self, slots: &[usize]) {
        for s in slots {
            self.tokens.remove(s);
        }
    }

    fn encode(&mut self, mel: &MelBatch, slots: &[usize]) -> Vec<WindowKey> {
        let per = mel.data.len() / mel.rows.max(1);
        slots
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let key = sha20(&f16_bytes(&mel.data[i * per..(i + 1) * per]));
                self.window.insert(s, key.clone());
                key
            })
            .collect()
    }

    fn window(&self, slot: usize) -> Result<WindowKey> {
        self.window
            .get(&slot)
            .cloned()
            .ok_or_else(|| eyre!("slot {slot} decodes with no encoded window"))
    }

    /// Writes `token` at `pos` of the slot's KV, as an append at the slot's length or an
    /// overwrite of positions a verify step fed but didn't keep, and returns the key of
    /// the logits after it
    fn feed(&mut self, slot: usize, pos: i32, token: u32) -> Result<(WindowKey, u64)> {
        let window = self.window(slot)?;
        let t = self.tokens.entry(slot).or_default();
        let pos = usize::try_from(pos)?;
        ensure!(
            pos <= t.len(),
            "slot {slot} fed at position {pos}, past the {} positions its KV holds",
            t.len()
        );
        t.truncate(pos);
        t.push(token);
        Ok((window, prefix_hash(t)))
    }
}

/// [`FakeModel`] that also records each result into an [`Oracle`]
struct Recorder {
    inner: FakeModel,
    oracle: Option<Rc<RefCell<Oracle>>>,
    track: SlotTrack,
}

impl Recorder {
    fn new(inner: FakeModel, oracle: Option<Rc<RefCell<Oracle>>>) -> Self {
        Self {
            inner,
            oracle,
            track: SlotTrack::default(),
        }
    }
}

/// The `[actual_len, cap]` rows of DTW batch row `bi` that the DP reads
fn dtw_region(batch: &DtwBatch, m: MatricesView<'_>, bi: usize) -> Vec<f32> {
    let [_, t_pad, width] = m.shape;
    let (len, cap) = (batch.actual_lens[bi], batch.audio_caps[bi].min(width));
    let mut out = Vec::with_capacity(len * cap);
    for r in 0..len.min(t_pad) {
        let base = (bi * t_pad + r) * width;
        out.extend_from_slice(&m.data[base..base + cap]);
    }
    out
}

fn dtw_key(batch: &DtwBatch, window: WindowKey, bi: usize) -> (WindowKey, Vec<u32>, usize) {
    let tokens = batch.row(bi)[..batch.actual_lens[bi]].to_vec();
    (window, tokens, batch.audio_caps[bi])
}

impl Model for Recorder {
    fn reset_kv(&mut self, slots: &[usize]) -> Result<()> {
        self.track.reset(slots);
        self.inner.reset_kv(slots)
    }

    fn encode(&mut self, mel: &MelBatch, slots: &[usize]) -> Result<()> {
        let keys = self.track.encode(mel, slots);
        if let Some(o) = &self.oracle {
            o.borrow_mut().windows.extend(keys);
        }
        self.inner.encode(mel, slots)
    }

    fn prefill(&mut self, tokens: &[u32], slots: &[usize]) -> Result<Logits> {
        let logits = self.inner.prefill(tokens, slots)?;
        let per = tokens.len() / slots.len();
        for (i, &s) in slots.iter().enumerate() {
            let row = tokens[i * per..(i + 1) * per].to_vec();
            self.track.tokens.insert(s, row.clone());
            if let Some(o) = &self.oracle {
                let key = (self.track.window(s)?, row);
                o.borrow_mut().prefill.insert(key, logits.row(i).to_vec());
            }
        }
        Ok(logits)
    }

    fn decode_step(&mut self, step: &DecodeStep) -> Result<Logits> {
        let logits = self.inner.decode_step(step)?;
        for i in 0..step.rows {
            let key = self
                .track
                .feed(step.slots[i], step.positions[i], step.tokens[i])?;
            if let Some(o) = &self.oracle {
                o.borrow_mut().logits.insert(key, logits.row(i).to_vec());
            }
        }
        Ok(logits)
    }

    fn dtw_prep(&mut self, batch: &DtwBatch, xa_slots: &[usize]) -> Result<Matrices> {
        let m = self.inner.dtw_prep(batch, xa_slots)?;
        if let Some(o) = &self.oracle {
            for (bi, &xa) in xa_slots.iter().enumerate().take(batch.rows()) {
                let key = dtw_key(batch, self.track.window(xa)?, bi);
                let region = dtw_region(batch, m.view(), bi);
                o.borrow_mut().dtw.insert(key, region);
            }
        }
        Ok(m)
    }

    fn dtw_launch(&mut self, batch: &DtwBatch, xa_slots: &[usize]) -> Result<Box<dyn DtwMatrices>> {
        let m = self.dtw_prep(batch, xa_slots)?;
        Ok(if self.inner.0.borrow().defer_dtw {
            Box::new(InFlight(m))
        } else {
            Box::new(ReadyMatrices(m))
        })
    }

    fn embed(&mut self, audio: &[f32]) -> Result<[f32; EMBED_DIM]> {
        let e = self.inner.embed(audio)?;
        if let Some(o) = &self.oracle {
            o.borrow_mut().embed.insert(sha20(&f32_bytes(audio)), e);
        }
        Ok(e)
    }

    fn reset_embeds(&mut self) -> Result<()> {
        self.inner.reset_embeds()
    }

    fn prefetch_embed(&mut self, audio: &Arc<[f32]>, start: usize, len: usize) -> Result<()> {
        self.inner.prefetch_embed(audio, start, len)
    }

    fn embed_crop(
        &mut self,
        audio: &Arc<[f32]>,
        start: usize,
        len: usize,
    ) -> Result<[f32; EMBED_DIM]> {
        // the inner model checks the prefetch, then calls its own embed
        let e = self.inner.embed_crop(audio, start, len)?;
        if let Some(o) = &self.oracle {
            o.borrow_mut()
                .embed
                .insert(sha20(&f32_bytes(&audio[start..start + len])), e);
        }
        Ok(e)
    }
}

/// Answers every call from an [`Oracle`]: an ideal model whose outputs depend only on
/// each slot's window and token prefix
struct OracleModel {
    oracle: Rc<RefCell<Oracle>>,
    track: SlotTrack,
    defer_dtw: bool,
    buckets: Vec<usize>,
    /// [`Model::verify_capacity`], small in one case so wide drafts fall back
    capacity: usize,
}

/// Verify logits, `None` for a prefix the reference never ran
struct OracleLogits(Vec<Option<Vec<f16>>>);

impl StepLogits for OracleLogits {
    fn rows(&self) -> usize {
        self.0.len()
    }

    fn row(&self, r: usize) -> Result<&[f16]> {
        match self.0.get(r) {
            Some(Some(row)) => Ok(row),
            Some(None) => bail!(
                "logits row {r} was read, but the reference never ran its token prefix: a \
                 position after a rejected draft reached a decision"
            ),
            None => bail!("logits row {r} of {}", self.0.len()),
        }
    }

    fn landed(&self) -> Result<usize> {
        Ok(self.0.len())
    }

    fn complete(&self) -> Result<()> {
        Ok(())
    }
}

impl Model for OracleModel {
    fn verify_capacity(&self) -> usize {
        self.capacity
    }

    fn reset_kv(&mut self, slots: &[usize]) -> Result<()> {
        self.track.reset(slots);
        Ok(())
    }

    fn encode(&mut self, mel: &MelBatch, slots: &[usize]) -> Result<()> {
        for key in self.track.encode(mel, slots) {
            ensure!(
                self.oracle.borrow().windows.contains(&key),
                "encoder input {key} never ran in the reference"
            );
        }
        Ok(())
    }

    fn prefill(&mut self, tokens: &[u32], slots: &[usize]) -> Result<Logits> {
        let per = tokens.len() / slots.len();
        let o = self.oracle.borrow();
        let mut data = Vec::new();
        for (i, &s) in slots.iter().enumerate() {
            let row = tokens[i * per..(i + 1) * per].to_vec();
            self.track.tokens.insert(s, row.clone());
            let key = (self.track.window(s)?, row);
            let l = o
                .prefill
                .get(&key)
                .ok_or_else(|| eyre!("prefill of slot {s} never ran in the reference"))?;
            data.extend_from_slice(l);
        }
        Logits::new(slots.len(), data.len() / slots.len(), data)
    }

    fn decode_step(&mut self, step: &DecodeStep) -> Result<Logits> {
        let n = step.rows;
        ensure!(
            step.bucket == get_bucket(n, &self.buckets)
                && step.slots[n..].iter().all(|&s| s == step.trash_slot),
            "decode step padding"
        );
        let mut data = Vec::new();
        for i in 0..n {
            let key = self
                .track
                .feed(step.slots[i], step.positions[i], step.tokens[i])?;
            let o = self.oracle.borrow();
            let l = o.logits.get(&key).ok_or_else(|| {
                eyre!(
                    "one-token step row {i} (slot {}, position {}) has a prefix the \
                     reference never ran",
                    step.slots[i],
                    step.positions[i]
                )
            })?;
            data.extend_from_slice(l);
        }
        Logits::new(n, data.len() / n.max(1), data)
    }

    fn launch_verify<'a>(
        &'a mut self,
        step: &VerifyStep,
        during: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Box<dyn StepLogits + 'a>> {
        let real = step.real_tokens();
        ensure!(
            step.starts.len() == step.rows + 1
                && step.slots.len() == step.rows
                && step.starts[0] == 0
                && step.starts.windows(2).all(|w| w[1] > w[0])
                && step.tokens.len() == get_bucket(real, &self.buckets)
                && step.positions.len() == step.tokens.len()
                && step.tokens[real..].iter().all(|&t| t == 0)
                && step.positions[real..].iter().all(|&p| p == 0),
            "verify step layout: {step:?}"
        );
        let mut rows = Vec::with_capacity(real);
        let mut seen = HashSet::new();
        for r in 0..step.rows {
            let s = step.slots[r];
            ensure!(s < step.trash_slot && seen.insert(s), "verify slot {s}");
            let start = step.starts[r];
            // a row starts at or before its slot's KV length (past it lie positions an
            // earlier verify step fed but didn't keep) and runs on consecutive positions;
            // a wrong start shows as a prefix the oracle doesn't hold
            let len = self.track.tokens.get(&s).map_or(0, Vec::len);
            let first = usize::try_from(step.positions[start])?;
            ensure!(
                first <= len,
                "slot {s} verify row starts at position {first}, past the {len} its KV holds"
            );
            for j in start..step.starts[r + 1] {
                let pos = step.positions[j];
                ensure!(
                    usize::try_from(pos)? == first + (j - start),
                    "slot {s} verify token {} at position {pos}, the row starts at {first}",
                    j - start
                );
                let key = self.track.feed(s, pos, step.tokens[j])?;
                if step.logit_row(j).is_some() {
                    rows.push(self.oracle.borrow().logits.get(&key).cloned());
                }
            }
        }
        ensure!(
            rows.len() == step.need.len()
                && step.need.iter().all(|&j| j < real)
                && step.head_rows == get_bucket(step.need.len(), &self.buckets),
            "verify step needs logits of {:?} over {} LM-head rows",
            step.need,
            step.head_rows
        );

        during()?;
        Ok(Box::new(OracleLogits(rows)))
    }

    fn dtw_prep(&mut self, batch: &DtwBatch, xa_slots: &[usize]) -> Result<Matrices> {
        let (rows, t_pad, width) = (batch.rows(), batch.t_pad, batch.max_audio_cap());
        let mut data = vec![0.0f32; rows * t_pad * width];
        let o = self.oracle.borrow();
        for (bi, &xa) in xa_slots.iter().enumerate().take(rows) {
            let key = dtw_key(batch, self.track.window(xa)?, bi);
            let region = o
                .dtw
                .get(&key)
                .ok_or_else(|| eyre!("DTW row {bi} never ran in the reference"))?;
            let cap = batch.audio_caps[bi].min(width);
            for (r, src) in region.chunks(cap.max(1)).enumerate() {
                let base = (bi * t_pad + r) * width;
                data[base..base + src.len()].copy_from_slice(src);
            }
        }
        Ok(Matrices {
            shape: [rows, t_pad, width],
            data,
        })
    }

    fn dtw_launch(&mut self, batch: &DtwBatch, xa_slots: &[usize]) -> Result<Box<dyn DtwMatrices>> {
        let m = self.dtw_prep(batch, xa_slots)?;
        Ok(if self.defer_dtw {
            Box::new(InFlight(m))
        } else {
            Box::new(ReadyMatrices(m))
        })
    }

    fn embed(&mut self, audio: &[f32]) -> Result<[f32; EMBED_DIM]> {
        self.oracle
            .borrow()
            .embed
            .get(&sha20(&f32_bytes(audio)))
            .copied()
            .ok_or_else(|| {
                eyre!(
                    "embedding of {} samples never ran in the reference",
                    audio.len()
                )
            })
    }
}

/// Replays each file of `name` with speculation off (the reference replay, recording an
/// oracle), then on against the oracle, and checks that every window emits the same
/// tokens and every file the same outputs
fn spec_proof(
    name: &str,
    config: Config,
    spec: SpecConfig,
    defer_dtw: bool,
    capacity: usize,
) -> Result<SpecStats> {
    let mut total = SpecStats::default();
    let Some(set) = TraceSet::open(name) else {
        return Ok(total);
    };
    let set = Rc::new(set);
    let tables = Arc::new(DecoderTables::new(Tokenizer::multilingual()?)?);
    let checks: Rc<RefCell<BTreeMap<String, usize>>> = Rc::default();
    for stem in set.stems()? {
        let oracle = Rc::new(RefCell::new(Oracle::default()));
        replay_recorded(
            &set,
            &stem,
            &config,
            &tables,
            &checks,
            false,
            Some(oracle.clone()),
        )?;
        let want = std::mem::take(&mut oracle.borrow_mut().tokens);

        let events = set.events(&stem)?;
        let result = set.result(&stem)?;
        let samples = result["samples"]
            .as_u64()
            .ok_or_else(|| eyre!("no samples"))? as usize;
        let audio: Arc<[f32]> = file_audio(&set, &events, samples)?.into();

        let mel_oracle = oracle.clone();
        let model = OracleModel {
            oracle: oracle.clone(),
            track: SlotTrack::default(),
            defer_dtw,
            buckets: captured_buckets(config.bs),
            capacity,
        };
        let mut trn = Transcriber::new(
            Config {
                spec: Some(spec),
                ..config.clone()
            },
            model,
            Box::new(move |a: &[f32]| {
                mel_oracle
                    .borrow()
                    .mel
                    .get(&sha20(&f32_bytes(a)))
                    .cloned()
                    .ok_or_else(|| eyre!("mel of {} samples never ran in the reference", a.len()))
            }),
            tables.clone(),
        )?;
        let got: Rc<RefCell<WindowTokens>> = Rc::default();
        let probe_got = got.clone();
        trn.set_probe(Box::new(move |event| {
            if let Event::UpdateState { slot, text, token } = event {
                probe_got.borrow_mut().record(slot, text, *token);
            }
        }));

        let out = trn.transcribe(&audio, &mut PyRandom::new(0))?;
        let tokens = got.borrow().check_same(&want, &stem)?;
        ensure!(
            out.text == result["text"].as_str().unwrap_or_default(),
            "{stem}: spec text differs\npython {:?}\nrust   {:?}",
            result["text"],
            out.text
        );
        ensure!(
            json_eq(&serde_json::to_value(&out.segments)?, &result["segments"]),
            "{stem}: spec segments differ"
        );
        let raw = result["raw_chunks"]
            .as_array()
            .ok_or_else(|| eyre!("raw_chunks"))?;
        ensure!(
            raw.len() == out.raw_chunks.len(),
            "{stem}: spec chunk count"
        );
        for (py, mine) in raw.iter().zip(&out.raw_chunks) {
            let diffs = result_diffs(py, Some(mine));
            ensure!(
                diffs.is_empty(),
                "{stem}: spec raw_chunks {}: {diffs:?}",
                mine.id
            );
        }

        let stats = trn.spec_stats();
        eprintln!(
            "{stem}: {tokens} tokens in {} windows match with speculation; {}",
            want.0.len(),
            stats.summary()
        );
        add_stats(&mut total, stats);
    }

    Ok(total)
}

fn add_stats(total: &mut SpecStats, s: &SpecStats) {
    total.verify_steps += s.verify_steps;
    total.forced_kept += s.forced_kept;
    total.carry_kept += s.carry_kept;
    total.ngram_kept += s.ngram_kept;
    total.ngram_fed += s.ngram_fed;
    total.full_steps += s.full_steps;
}

/// Runs [`spec_proof`] over a spread of configs; `real` sets must also keep drafts of
/// every source, which the synthetic noise sets mostly never draft
fn spec_all(name: &str, config: Config, real: bool) -> Result<()> {
    let cases = [
        (
            SpecConfig {
                k: 1,
                max_rows: 64,
                carry: false,
                ngram: 0,
            },
            false,
            usize::MAX,
        ),
        (
            SpecConfig {
                k: 4,
                max_rows: 64,
                carry: true,
                ngram: 2,
            },
            true,
            usize::MAX,
        ),
        (
            SpecConfig {
                k: 7,
                max_rows: 64,
                carry: true,
                ngram: 7,
            },
            false,
            usize::MAX,
        ),
        (
            SpecConfig {
                k: 4,
                max_rows: 2,
                carry: true,
                ngram: 0,
            },
            true,
            usize::MAX,
        ),
        // a verify workspace of 8 tokens: wider drafts fall back to one token a row
        (
            SpecConfig {
                k: 4,
                max_rows: 64,
                carry: true,
                ngram: 2,
            },
            true,
            8,
        ),
    ];
    let mut total = SpecStats::default();
    let mut full_steps = 0;
    for (spec, defer, capacity) in cases {
        eprintln!("{name}: {spec:?}, deferred DTW {defer}, verify capacity {capacity}");
        let stats = spec_proof(name, config.clone(), spec, defer, capacity)?;
        full_steps += stats.full_steps;
        add_stats(&mut total, &stats);
    }

    // the proof means nothing unless verify steps kept drafts of each source
    if real && TraceSet::open(name).is_some() {
        ensure!(
            total.verify_steps > 0
                && total.forced_kept > 0
                && total.carry_kept > 0
                && total.ngram_kept > 0,
            "{name}: verify steps never kept a draft of each source: {total:?}"
        );
        ensure!(
            full_steps > 0,
            "{name}: no step overfilled the 8-token verify workspace"
        );
    }
    Ok(())
}

#[test]
fn spec_short3() -> Result<()> {
    spec_all("short3", Config::default(), true)
}

#[test]
fn spec_e21c600() -> Result<()> {
    spec_all("e21c600", Config::default(), true)
}

#[test]
fn spec_synthetic_bs4() -> Result<()> {
    spec_all(
        &format!("{SYNTHETIC}/synth-bs4"),
        Config {
            bs: 4,
            ..Config::default()
        },
        false,
    )
}

#[test]
fn spec_synthetic_bs6_lockstep() -> Result<()> {
    spec_all(
        &format!("{SYNTHETIC}/synth-bs6-lockstep"),
        Config {
            bs: 6,
            ..Config::default()
        },
        false,
    )
}
