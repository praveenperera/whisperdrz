//! Replays every recorded `decode_batch` and `update_state_after_decode` call from the
//! reference traces through the Rust state machine, row by row, and compares the
//! returned token and the slot state afterwards with python's

use std::collections::BTreeMap;

use eyre::{Result, bail, ensure, eyre};
use half::f16;
use serde_json::{Map, Value};
use whisperdrz_fixtures::{ArrayData, Event, TraceSet, as_f64};
use whisperdrz_native::decode::{DecodeConfig, Decoder, DecoderTables, SlotState};
use whisperdrz_native::text::Tokenizer;

#[path = "support/state_json.rs"]
mod state_json;

use state_json::{opt_str, same, state_fields, state_from};

/// Snapshot fields the Rust state models, by python attribute name
const MODELED: [&str; 17] = [
    "language",
    "status",
    "offset",
    "cur",
    "last_timing",
    "token_count",
    "last_token_id",
    "recent_tokens",
    "forced_timing_before_newline",
    "force_timing_every_word",
    "force_timestamp_next",
    "predict_next",
    "stop_reason",
    "_window_max_timing",
    "_timing_count",
    "_line_cache",
    "meta",
];

fn snapshot(value: &Value) -> Result<&Map<String, Value>> {
    value
        .get("__state")
        .and_then(Value::as_object)
        .ok_or_else(|| eyre!("not a state snapshot: {value}"))
}

#[derive(Debug)]
struct Mismatch {
    stem: String,
    seq: u64,
    row: Option<usize>,
    field: String,
    python: String,
    rust: String,
}

impl std::fmt::Display for Mismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let row = self.row.map(|r| format!(" row {r}")).unwrap_or_default();
        write!(
            f,
            "{} seq {}{row}: {}: python {} rust {}",
            self.stem, self.seq, self.field, self.python, self.rust
        )
    }
}

#[derive(Default)]
struct Report {
    batches: usize,
    rows: usize,
    updates: usize,
    mismatches: Vec<Mismatch>,
    samples: usize,
    exact_ties: usize,
    min_margin: Option<(f32, String)>,
    division_sensitive: usize,
    /// How often each branch fired, inferred from decisions and state changes
    coverage: BTreeMap<String, usize>,
}

impl Report {
    fn mismatch(
        &mut self,
        stem: &str,
        seq: u64,
        row: Option<usize>,
        field: &str,
        python: String,
        rust: String,
    ) {
        self.mismatches.push(Mismatch {
            stem: stem.to_owned(),
            seq,
            row,
            field: field.to_owned(),
            python,
            rust,
        });
    }

    /// Compares the modeled fields of a Rust state with python's after-snapshot, and
    /// checks that python left every field Rust doesn't model unchanged
    fn compare_state(
        &mut self,
        stem: &str,
        seq: u64,
        row: Option<usize>,
        before: &Map<String, Value>,
        after: &Map<String, Value>,
        rust: &SlotState,
    ) {
        for (k, v) in state_fields(rust) {
            let py = after.get(k).unwrap_or(&Value::Null);
            if !same(py, &v) {
                self.mismatch(stem, seq, row, k, py.to_string(), v.to_string());
            }
        }

        for (k, v) in after {
            if MODELED.contains(&k.as_str()) && k != "meta" {
                continue;
            }

            let b = before.get(k).unwrap_or(&Value::Null);
            if !same(b, v) {
                self.mismatch(
                    stem,
                    seq,
                    row,
                    &format!("unmodeled {k} changed"),
                    b.to_string(),
                    v.to_string(),
                );
            }
        }
    }
}

fn logits_rows(set: &TraceSet, ev: &Event) -> Result<Vec<Vec<f16>>> {
    let arr = set.array(ev.arg("logits")?)?;
    let ArrayData::F16(data) = arr.data else {
        bail!("logits are {}, expected float16", arr.dtype)
    };
    let v = *arr.shape.last().ok_or_else(|| eyre!("scalar logits"))?;
    let rows = arr.shape[0];
    // [BS, 1, V] keeps the last position, which is the only one
    let per_row = data.len() / rows;
    Ok((0..rows)
        .map(|r| data[r * per_row + per_row - v..(r + 1) * per_row].to_vec())
        .collect())
}

fn check_settings(ev: &Event) -> Result<()> {
    let config = DecodeConfig::default();
    ensure!(
        as_f64(ev.arg("temperature")?) == Some(config.temperature),
        "seq {}: temperature",
        ev.seq
    );
    for (k, want) in [
        ("allow_tags", config.allow_tags),
        ("allow_asterisks", config.allow_asterisks),
        ("speaker_tags", config.speaker_tags),
    ] {
        ensure!(
            ev.arg(k)?.as_bool() == Some(want),
            "seq {}: {k} is {}",
            ev.seq,
            ev.arg(k)?
        );
    }

    Ok(())
}

fn replay_batch(
    set: &TraceSet,
    stem: &str,
    ev: &Event,
    decoder: &Decoder,
    report: &mut Report,
) -> Result<()> {
    check_settings(ev)?;
    let logits = logits_rows(set, ev)?;
    let before = ev
        .arg("states")?
        .as_array()
        .ok_or_else(|| eyre!("states is not a list"))?;
    let after = ev
        .after_arg("states")?
        .as_array()
        .ok_or_else(|| eyre!("after.states is not a list"))?;
    let ret = ev
        .ret()?
        .as_array()
        .ok_or_else(|| eyre!("ret is not a list"))?;
    let last_ids = ev
        .arg("last_token_ids")?
        .as_array()
        .ok_or_else(|| eyre!("last_token_ids"))?;
    let counts = ev
        .arg("token_counts")?
        .as_array()
        .ok_or_else(|| eyre!("token_counts"))?;
    let slots = ev
        .arg("slot_indices")?
        .as_array()
        .ok_or_else(|| eyre!("slot_indices"))?;
    let windows = ev
        .arg("all_tokens_gpu")?
        .get("__rows")
        .and_then(Value::as_object)
        .ok_or_else(|| eyre!("no __rows"))?;
    ensure!(
        logits.len() == before.len() && ret.len() == before.len(),
        "seq {}: batch sizes differ",
        ev.seq
    );

    report.batches += 1;
    for (r, row_logits) in logits.iter().enumerate() {
        let snap = snapshot(&before[r])?;
        let mut state = state_from(snap)?;
        ensure!(
            last_ids[r].as_u64() == Some(state.last_token_id as u64),
            "seq {} row {r}: last_token_ids",
            ev.seq
        );
        ensure!(
            counts[r].as_u64() == Some(state.token_count as u64),
            "seq {} row {r}: token_counts",
            ev.seq
        );

        let window: Option<Vec<u32>> = match Decoder::repetition_window(state.token_count) {
            Some(_) => {
                let key = slots[r].to_string();
                let w = windows
                    .get(&key)
                    .and_then(Value::as_array)
                    .ok_or_else(|| eyre!("seq {}: no window for slot {key}", ev.seq))?;
                Some(
                    w.iter()
                        .map(|t| {
                            t.as_u64()
                                .map(|t| t as u32)
                                .ok_or_else(|| eyre!("bad window token"))
                        })
                        .collect::<Result<_>>()?,
                )
            }
            None => None,
        };

        let decision = decoder.decode_row(row_logits, &mut state, window.as_deref())?;
        report.rows += 1;

        let want = ret[r]
            .get("__tuple")
            .and_then(Value::as_array)
            .ok_or_else(|| eyre!("ret row is not a tuple"))?;
        let (want_text, want_id) = (
            want[0].as_str().unwrap_or_default(),
            want[1].as_u64().unwrap_or(u64::MAX),
        );
        if want_text != decision.text {
            report.mismatch(
                stem,
                ev.seq,
                Some(r),
                "ret.text",
                format!("{want_text:?}"),
                format!("{:?}", decision.text),
            );
        }

        if want_id != decision.token as u64 {
            let detail = format!("{} {:?}", decision.token, decision.samples);
            report.mismatch(
                stem,
                ev.seq,
                Some(r),
                "ret.token",
                want_id.to_string(),
                detail,
            );
        }

        report.compare_state(stem, ev.seq, Some(r), snap, snapshot(&after[r])?, &state);
        let mut hit = |k: String| *report.coverage.entry(k).or_default() += 1;
        if decision.samples.is_empty() {
            hit(format!("forced {:?}", decision.text));
        }

        for s in decision.samples.iter().skip(1) {
            hit(format!("stage {:?}", s.stage));
        }

        if let Some(reason) = state
            .stop_reason
            .as_deref()
            .filter(|_| opt_str(snap, "stop_reason").ok().flatten().is_none())
        {
            hit(format!(
                "stop {}",
                reason.split(':').next().unwrap_or(reason)
            ));
        }

        if state.predict_next.is_some() {
            hit(format!("defer {:?}", state.predict_next));
        }

        if state.force_timestamp_next {
            hit("force_timestamp_next set".to_owned());
        }

        if window.is_some() {
            hit("repetition check ran".to_owned());
        }

        for s in &decision.samples {
            report.samples += 1;
            report.division_sensitive += s.division_sensitive as usize;
            if let Some(m) = s.argmax.margin() {
                if m == 0.0 {
                    report.exact_ties += 1;
                }

                if report.min_margin.as_ref().is_none_or(|(best, _)| m < *best) {
                    let at = format!(
                        "{stem} seq {} row {r} {:?} token {}",
                        ev.seq, s.stage, s.argmax.token
                    );
                    report.min_margin = Some((m, at));
                }
            }
        }
    }

    Ok(())
}

fn replay_update(stem: &str, ev: &Event, report: &mut Report) -> Result<()> {
    let snap = snapshot(ev.arg("state")?)?;
    let mut state = state_from(snap)?;
    let newtext = ev
        .arg("newtext")?
        .as_str()
        .ok_or_else(|| eyre!("newtext"))?;
    let token = ev
        .arg("newtoken_id")?
        .as_u64()
        .ok_or_else(|| eyre!("newtoken_id"))? as u32;
    let skip_cur = ev
        .args
        .get("skip_cur")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    state.update_after_decode(newtext, token, skip_cur)?;
    report.updates += 1;
    report.compare_state(
        stem,
        ev.seq,
        None,
        snap,
        snapshot(ev.after_arg("state")?)?,
        &state,
    );
    Ok(())
}

fn replay(set_name: &str) -> Result<()> {
    let Some(set) = TraceSet::open(set_name) else {
        return Ok(());
    };
    let tables = DecoderTables::new(Tokenizer::multilingual()?)?;
    let decoder = Decoder::new(&tables, DecodeConfig::default())?;
    let mut report = Report::default();

    for stem in set.stems()? {
        for ev in set.events(&stem)? {
            match ev.ev.as_str() {
                "decode_batch" => replay_batch(&set, &stem, &ev, &decoder, &mut report)?,
                "update_state_after_decode" => replay_update(&stem, &ev, &mut report)?,
                _ => {}
            }
        }
    }

    eprintln!(
        "{set_name}: {} decode_batch calls, {} rows, {} update_state_after_decode calls, {} mismatches",
        report.batches,
        report.rows,
        report.updates,
        report.mismatches.len()
    );
    eprintln!(
        "{set_name}: {} argmaxes, {} decided by an exact tie (lowest id), {} sensitive to the penalty's reciprocal, smallest margin {:?}",
        report.samples, report.exact_ties, report.division_sensitive, report.min_margin
    );

    eprintln!("{set_name}: branches hit: {:?}", report.coverage);
    ensure!(
        report.rows > 0 && report.updates > 0,
        "{set_name}: nothing replayed"
    );
    if let Some(first) = report.mismatches.first() {
        bail!(
            "{set_name}: {} mismatches, first: {first}",
            report.mismatches.len()
        );
    }

    Ok(())
}

#[test]
fn replay_short3() -> Result<()> {
    replay("short3")
}

#[test]
fn replay_e21c600() -> Result<()> {
    replay("e21c600")
}
