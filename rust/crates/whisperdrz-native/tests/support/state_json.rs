//! Python state snapshots (the tracer's `{"__state": {...}}` bodies) to and from
//! [`SlotState`], shared by the replay and synthetic-case tests

use std::collections::BTreeMap;

use eyre::{Result, bail, eyre};
use serde_json::{Map, Value};
use whisperdrz_fixtures::as_f64;
use whisperdrz_native::decode::{SlotState, Status};

fn f64_field(s: &Map<String, Value>, k: &str) -> Result<f64> {
    s.get(k)
        .and_then(as_f64)
        .ok_or_else(|| eyre!("state field {k} is not a float"))
}

pub fn opt_str(s: &Map<String, Value>, k: &str) -> Result<Option<String>> {
    match s.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(v)) => Ok(Some(v.clone())),
        Some(v) => bail!("state field {k} is not a string: {v}"),
    }
}

pub fn state_from(s: &Map<String, Value>) -> Result<SlotState> {
    let str_field = |k: &str| opt_str(s, k)?.ok_or_else(|| eyre!("state field {k} missing"));
    let bool_field = |k: &str| {
        s.get(k)
            .and_then(Value::as_bool)
            .ok_or_else(|| eyre!("state field {k} is not a bool"))
    };
    let int_field = |k: &str| {
        s.get(k)
            .and_then(Value::as_u64)
            .ok_or_else(|| eyre!("state field {k} is not an int"))
    };

    // InferenceState.duration is the chunk audio length, which the snapshot keeps in meta
    let duration = s
        .get("meta")
        .and_then(|m| m.get("duration"))
        .and_then(as_f64);
    let recent = s["recent_tokens"]
        .as_array()
        .ok_or_else(|| eyre!("recent_tokens is not a list"))?
        .iter()
        .map(|t| {
            t.as_u64()
                .map(|t| t as u32)
                .ok_or_else(|| eyre!("bad recent token {t}"))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(SlotState {
        language: str_field("language")?,
        status: Status::parse(&str_field("status")?)?,
        offset: f64_field(s, "offset")?,
        duration: duration.ok_or_else(|| eyre!("state has no meta.duration"))?,
        cur: str_field("cur")?,
        last_timing: s.get("last_timing").and_then(as_f64),
        token_count: int_field("token_count")? as usize,
        last_token_id: int_field("last_token_id")? as u32,
        recent_tokens: recent,
        forced_timing_before_newline: bool_field("forced_timing_before_newline")?,
        force_timing_every_word: bool_field("force_timing_every_word")?,
        force_timestamp_next: bool_field("force_timestamp_next")?,
        predict_next: opt_str(s, "predict_next")?,
        stop_reason: opt_str(s, "stop_reason")?,
        window_max_timing: f64_field(s, "_window_max_timing")?,
        timing_count: int_field("_timing_count")? as usize,
        line_cache: s
            .get("_line_cache")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned(),
    })
}

/// The Rust state's fields as python values, for comparison against an `after` snapshot
pub fn state_fields(s: &SlotState) -> BTreeMap<&'static str, Value> {
    let f = |x: f64| serde_json::Number::from_f64(x).map_or(Value::Null, Value::Number);
    BTreeMap::from([
        ("language", Value::from(s.language.clone())),
        ("status", Value::from(s.status.as_str())),
        ("offset", f(s.offset)),
        ("cur", Value::from(s.cur.clone())),
        ("last_timing", s.last_timing.map_or(Value::Null, f)),
        ("token_count", Value::from(s.token_count)),
        ("last_token_id", Value::from(s.last_token_id)),
        ("recent_tokens", Value::from(s.recent_tokens.clone())),
        (
            "forced_timing_before_newline",
            Value::from(s.forced_timing_before_newline),
        ),
        (
            "force_timing_every_word",
            Value::from(s.force_timing_every_word),
        ),
        ("force_timestamp_next", Value::from(s.force_timestamp_next)),
        (
            "predict_next",
            s.predict_next.clone().map_or(Value::Null, Value::from),
        ),
        (
            "stop_reason",
            s.stop_reason.clone().map_or(Value::Null, Value::from),
        ),
        ("_window_max_timing", f(s.window_max_timing)),
        ("_timing_count", Value::from(s.timing_count)),
        ("_line_cache", Value::from(s.line_cache.clone())),
    ])
}

/// Numbers compare by value (python ints and floats both decode to JSON numbers)
pub fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        _ => a == b,
    }
}
