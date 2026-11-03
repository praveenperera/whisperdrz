//! Hand-built slot states for the `decode_batch` and `update_state_after_decode`
//! branches the reference traces never reach (language step, repetition stop, forced
//! bracket and asterisk closes, the EOT resample, deferred tokens, an all-blocked
//! timing row, ...), with python's results from `tests/data/decode_cases.json`
//! (generated on the CPU by `rust/tools/dump_decode_reference.py`)

use eyre::{Result, bail, eyre};
use half::f16;
use serde_json::Value;
use whisperdrz_native::decode::{DecodeConfig, Decoder, DecoderTables};
use whisperdrz_native::text::Tokenizer;

#[path = "support/state_json.rs"]
mod state_json;

use state_json::{same, state_fields, state_from};

fn load() -> Result<Value> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/decode_cases.json");
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

/// The generator's `synth_logits`: an integer hash spread over [-4, 4) in 1/128 steps,
/// or a constant fill, then explicit `[token, value]` boosts
fn synth_logits(spec: &Value, v: usize) -> Result<Vec<f16>> {
    let mut row: Vec<f16> = match spec.get("fill").and_then(Value::as_f64) {
        Some(fill) => vec![f16::from_f64(fill); v],
        None => {
            let seed = spec.get("seed").and_then(Value::as_u64).unwrap_or(0);
            (0..v as u64)
                .map(|j| {
                    let h = (j * 2_654_435_761 + seed * 40_503) % (1 << 32);
                    f16::from_f64(((h >> 22) as f64 - 512.0) / 128.0)
                })
                .collect()
        }
    };

    for b in spec
        .get("boosts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let t = b[0].as_u64().ok_or_else(|| eyre!("boost token"))? as usize;
        row[t] = f16::from_f64(b[1].as_f64().ok_or_else(|| eyre!("boost value"))?);
    }

    Ok(row)
}

fn diff_state(
    name: &str,
    want: &Value,
    got: &whisperdrz_native::decode::SlotState,
    out: &mut Vec<String>,
) {
    for (k, v) in state_fields(got) {
        let py = want.get(k).unwrap_or(&Value::Null);
        if !same(py, &v) {
            out.push(format!("{name}: {k}: python {py} rust {v}"));
        }
    }
}

#[test]
fn synthetic_cases_match_python() -> Result<()> {
    let cases = load()?;
    let tables = DecoderTables::new(Tokenizer::multilingual()?)?;
    let decoder = Decoder::new(&tables, DecodeConfig::default())?;
    let mut diffs = Vec::new();

    let decode = cases["decode"]
        .as_array()
        .ok_or_else(|| eyre!("decode cases"))?;
    for case in decode {
        let name = case["name"].as_str().unwrap_or("?");
        let mut state = state_from(
            case["state"]
                .as_object()
                .ok_or_else(|| eyre!("{name}: state"))?,
        )?;
        let logits = synth_logits(&case["logits"], tables.n_vocab())?;
        let window: Option<Vec<u32>> = match &case["window"] {
            Value::Null => None,
            w => Some(
                w.as_array()
                    .ok_or_else(|| eyre!("window"))?
                    .iter()
                    .map(|t| t.as_u64().unwrap_or(0) as u32)
                    .collect(),
            ),
        };

        let d = decoder.decode_row(&logits, &mut state, window.as_deref())?;
        let (text, id) = (case["ret"][0].as_str(), case["ret"][1].as_u64());
        if text != Some(d.text.as_str()) || id != Some(d.token as u64) {
            diffs.push(format!(
                "{name}: ret python {} rust ({:?}, {})",
                case["ret"], d.text, d.token
            ));
        }

        diff_state(name, &case["after"], &state, &mut diffs);
    }

    let update = cases["update"]
        .as_array()
        .ok_or_else(|| eyre!("update cases"))?;
    for case in update {
        let name = case["name"].as_str().unwrap_or("?");
        let mut state = state_from(
            case["state"]
                .as_object()
                .ok_or_else(|| eyre!("{name}: state"))?,
        )?;
        let newtext = case["newtext"]
            .as_str()
            .ok_or_else(|| eyre!("{name}: newtext"))?;
        let token = case["token"]
            .as_u64()
            .ok_or_else(|| eyre!("{name}: token"))? as u32;
        state.update_after_decode(newtext, token, false)?;
        diff_state(name, &case["after"], &state, &mut diffs);
    }

    if !diffs.is_empty() {
        bail!("{} differences:\n{}", diffs.len(), diffs.join("\n"));
    }

    eprintln!(
        "{} decode and {} update cases match",
        decode.len(),
        update.len()
    );
    Ok(())
}

#[test]
fn unported_settings_are_rejected() {
    let base = DecodeConfig::default();
    assert!(base.validate().is_ok());
    for config in [
        DecodeConfig {
            temperature: 0.7,
            ..base
        },
        DecodeConfig {
            allow_tags: true,
            ..base
        },
        DecodeConfig {
            allow_asterisks: true,
            ..base
        },
        DecodeConfig {
            speaker_tags: false,
            ..base
        },
    ] {
        assert!(config.validate().is_err(), "{config:?} should be rejected");
    }
}
