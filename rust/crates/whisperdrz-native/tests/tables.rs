//! Compares every `DecoderState` / `inference_utils` table and the per-character string
//! properties with python's, from `tests/data/decode_tables.json` (written on the CPU by
//! `rust/tools/dump_decode_reference.py`)

use std::collections::{BTreeMap, BTreeSet};

use eyre::{Result, bail, eyre};
use serde_json::Value;
use sha2::{Digest, Sha256};
use whisperdrz_native::decode::DecoderTables;
use whisperdrz_native::decode::text_rules::is_word_char;
use whisperdrz_native::text::vocab::token_script;
use whisperdrz_native::text::{Tokenizer, py};

fn load() -> Result<Value> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/decode_tables.json");
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

fn digest(ids: &[u32]) -> String {
    let joined = ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
    format!("{:x}", Sha256::digest(joined.as_bytes()))
}

fn ints(v: &Value) -> Result<Vec<u32>> {
    v.as_array()
        .ok_or_else(|| eyre!("not a list: {v}"))?
        .iter()
        .map(|x| {
            x.as_u64()
                .map(|x| x as u32)
                .ok_or_else(|| eyre!("not an int: {x}"))
        })
        .collect()
}

fn expand_runs(v: &Value) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    for r in v.as_array().ok_or_else(|| eyre!("runs are not a list"))? {
        let r = ints(r)?;
        out.extend(r[0]..=r[1]);
    }

    Ok(out)
}

/// Differences found, one line each
#[derive(Default)]
struct Diffs(Vec<String>);

impl Diffs {
    /// An unordered set: `{"runs": ...}` or `{"n", "sha256"}`
    fn set(&mut self, name: &str, want: &Value, got: impl IntoIterator<Item = u32>) -> Result<()> {
        let got: Vec<u32> = got
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        if let Some(runs) = want.get("runs") {
            let want: BTreeSet<u32> = expand_runs(runs)?.into_iter().collect();
            let got: BTreeSet<u32> = got.into_iter().collect();
            if want != got {
                let missing: Vec<_> = want.difference(&got).take(10).collect();
                let extra: Vec<_> = got.difference(&want).take(10).collect();
                self.0
                    .push(format!("{name}: missing {missing:?} extra {extra:?}"));
            }

            return Ok(());
        }

        let n = want["n"].as_u64().ok_or_else(|| eyre!("{name}: no n"))? as usize;
        if n != got.len() || want["sha256"].as_str() != Some(&digest(&got)) {
            self.0.push(format!(
                "{name}: python {n} ids, rust {} (or digest differs)",
                got.len()
            ));
        }

        Ok(())
    }

    /// A list: `{"ids", "ordered"}` or `{"n", "sha256", "ordered"}`
    fn list(&mut self, name: &str, want: &Value, got: &[u32]) -> Result<()> {
        let mut got = got.to_vec();
        if want["ordered"].as_bool() != Some(true) {
            got.sort_unstable();
        }

        if let Some(ids) = want.get("ids") {
            let ids = ints(ids)?;
            if ids != got {
                self.0.push(format!("{name}: python {ids:?} rust {got:?}"));
            }

            return Ok(());
        }

        let n = want["n"].as_u64().ok_or_else(|| eyre!("{name}: no n"))? as usize;
        if n != got.len() || want["sha256"].as_str() != Some(&digest(&got)) {
            self.0.push(format!(
                "{name}: python {n} ids, rust {} (or digest/order differs)",
                got.len()
            ));
        }

        Ok(())
    }
}

#[test]
fn decoder_tables_match_python() -> Result<()> {
    let want = load()?;
    let ds = DecoderTables::new(Tokenizer::multilingual()?)?;
    let mut diffs = Diffs::default();

    if want["n_vocab"].as_u64() != Some(ds.n_vocab() as u64) {
        diffs.0.push(format!(
            "n_vocab: python {} rust {}",
            want["n_vocab"],
            ds.n_vocab()
        ));
    }

    let mut h = Sha256::new();
    for (_, t) in ds.vocab.iter() {
        h.update(t.as_bytes());
        h.update([0xff]);
    }

    if want["vocab_sha256"].as_str() != Some(&format!("{:x}", h.finalize())) {
        diffs.0.push("vocab_list digest differs".to_owned());
    }

    for pair in want["lossy_tokens"]
        .as_array()
        .ok_or_else(|| eyre!("lossy_tokens"))?
    {
        let id = pair[0].as_u64().ok_or_else(|| eyre!("lossy id"))? as u32;
        if pair[1].as_str() != Some(ds.vocab.text(id)) {
            diffs.0.push(format!(
                "decode([{id}]): python {} rust {:?}",
                pair[1],
                ds.vocab.text(id)
            ));
        }
    }

    let ids = ds.ids;
    let rust_chars = BTreeMap::from([
        ("[", ids.open_bracket),
        ("]", ids.close_bracket),
        ("(", ids.open_paren),
        (")", ids.close_paren),
        ("<", ids.less),
        (">", ids.greater),
        (" ", ids.space),
        ("0", ids.zero),
        ("\n", ids.newline),
        ("*", ids.asterisk),
    ]);
    for (c, id) in want["single_char_ids"]
        .as_object()
        .ok_or_else(|| eyre!("single_char_ids"))?
    {
        if id.as_u64() != rust_chars.get(c.as_str()).map(|&t| t as u64) {
            diffs.0.push(format!("single_char_ids[{c:?}]: python {id}"));
        }
    }

    let lists = &ds.lists;
    let punct_chars = ["!", "?", ",", "-", ".", "…", "'"];
    let mut rust_lists: BTreeMap<String, Vec<u32>> = BTreeMap::from([
        ("timings".to_owned(), ds.timings.clone()),
        ("timing_token_ids".to_owned(), ds.timing_token_ids.clone()),
        ("space_starting".to_owned(), ds.space_starting.clone()),
        ("digit_starting".to_owned(), ds.digit_starting.clone()),
        ("space_tokens_only".to_owned(), ds.space_tokens_only.clone()),
        ("punctuation".to_owned(), ds.punctuation.clone()),
        (
            "bracket_candidates".to_owned(),
            ds.bracket_candidates.clone(),
        ),
        (
            "angle_tokens_blocked".to_owned(),
            lists.angle_tokens_blocked.clone(),
        ),
        ("initial_tokens".to_owned(), lists.initial_tokens.clone()),
        (
            "all_punct".to_owned(),
            ds.all_punct.iter().copied().collect(),
        ),
        ("lang_tokens".to_owned(), ds.tok.all_language_tokens()),
        ("dont_predict".to_owned(), lists.dont_predict.clone()),
        (
            "asterisk_tokens_unblockable".to_owned(),
            lists.asterisk_tokens_unblockable.clone(),
        ),
    ]);
    for (c, set) in punct_chars.iter().zip(&ds.punct_block_sets) {
        rust_lists.insert(format!("punct_block_set:{c}"), set.clone());
    }

    let want_lists = want["lists"].as_object().ok_or_else(|| eyre!("lists"))?;
    for (name, w) in want_lists {
        match rust_lists.get(name) {
            Some(got) => diffs.list(name, w, got)?,
            None => diffs.0.push(format!("{name}: not built in rust")),
        }
    }

    let m = &ds.masks;
    let rust_masks = [
        ("base_blocked", &m.base_blocked),
        ("base_extra_blocked", &m.base_extra_blocked),
        ("angle_blocked_mask", &m.angle_blocked_mask),
        ("asterisk_blocked_mask", &m.asterisk_blocked_mask),
        ("block_open_bracket", &m.block_open_bracket),
        ("block_close_bracket", &m.block_close_bracket),
        ("block_open_paren", &m.block_open_paren),
        ("block_close_paren", &m.block_close_paren),
        ("block_less_than", &m.block_less_than),
        ("block_greater_than", &m.block_greater_than),
        ("block_angle_consecutive", &m.block_angle_consecutive),
        ("block_asterisk", &m.block_asterisk),
        ("space_asterisk_token", &m.space_asterisk_token),
        ("block_contains_space", &m.block_contains_space),
        (
            "block_contains_close_bracket",
            &m.block_contains_close_bracket,
        ),
        (
            "block_contains_open_bracket",
            &m.block_contains_open_bracket,
        ),
        ("block_quote", &m.block_quote),
        ("block_timings", &m.block_timings),
        ("block_space_starting", &m.block_space_starting),
        ("block_digit_starting", &m.block_digit_starting),
        ("block_punct", &m.block_punct),
        ("block_newlines", &m.block_newlines),
        ("in_bracket_blocked", &m.in_bracket_blocked),
        ("in_angle_blocked", &m.in_angle_blocked),
        ("in_paren_blocked", &m.in_paren_blocked),
        ("after_timing_blocked", &m.after_timing_blocked),
        ("lang_enabled", &m.lang_enabled),
        ("space_starting_enabled", &m.space_starting_enabled),
        ("speaker_id_mask", &m.speaker_id_mask),
    ];
    let want_masks = want["masks"].as_object().ok_or_else(|| eyre!("masks"))?;
    if want_masks.len() != rust_masks.len() {
        diffs.0.push(format!(
            "python has {} masks, rust compares {}",
            want_masks.len(),
            rust_masks.len()
        ));
    }

    for (name, mask) in rust_masks {
        let w = want_masks
            .get(name)
            .ok_or_else(|| eyre!("python has no mask {name}"))?;
        diffs.set(name, w, mask.ones())?;
    }

    for (lang, w) in want["base_blocked_by_lang"]
        .as_object()
        .ok_or_else(|| eyre!("base_blocked_by_lang"))?
    {
        diffs.set(
            &format!("base_blocked[{lang}]"),
            w,
            ds.base_blocked(lang).ones(),
        )?;
    }

    let want_allowed = want["bracket_allowed"]
        .as_object()
        .ok_or_else(|| eyre!("bracket_allowed"))?;
    let rust_keys: BTreeSet<&str> = ds.bracket_allowed.keys().map(String::as_str).collect();
    let py_keys: BTreeSet<&str> = want_allowed.keys().map(String::as_str).collect();
    if rust_keys != py_keys {
        diffs.0.push(format!(
            "bracket_allowed keys: python {} rust {}",
            py_keys.len(),
            rust_keys.len()
        ));
    }

    for (prefix, w) in want_allowed {
        let got = ds.bracket_allowed.get(prefix).cloned().unwrap_or_default();
        if ints(w)? != got {
            diffs.0.push(format!(
                "bracket_allowed[{prefix:?}]: python {w} rust {got:?}"
            ));
        }
    }

    // string properties over every character that occurs in a token
    let chars_want = &want["chars"];
    let chars: Vec<char> = {
        let mut set = BTreeSet::new();
        for (_, t) in ds.vocab.iter() {
            set.extend(t.chars());
        }

        set.into_iter().collect()
    };
    diffs.set("chars", chars_want, chars.iter().map(|&c| c as u32))?;

    let props = &want["props"];
    let pick = |f: &dyn Fn(char) -> bool| {
        chars
            .iter()
            .filter(|&&c| f(c))
            .map(|&c| c as u32)
            .collect::<Vec<_>>()
    };
    diffs.set("isalnum", &props["isalnum"], pick(&py::is_alnum))?;
    diffs.set("isdigit", &props["isdigit"], pick(&py::is_digit))?;
    diffs.set("isspace", &props["isspace"], pick(&py::is_space))?;
    diffs.set(
        "isprintable",
        &props["isprintable"],
        pick(&py::is_printable),
    )?;
    diffs.set(
        "is_word_char",
        &props["is_word_char"],
        pick(&|c| is_word_char(Some(c))),
    )?;
    for (script, w) in props["script"].as_object().ok_or_else(|| eyre!("script"))? {
        let got = pick(&|c| token_script(&c.to_string()).unwrap_or("None") == script);
        diffs.set(&format!("script[{script}]"), w, got)?;
    }

    if !diffs.0.is_empty() {
        bail!(
            "{} tables differ from python:\n{}",
            diffs.0.len(),
            diffs.0.join("\n")
        );
    }

    Ok(())
}
