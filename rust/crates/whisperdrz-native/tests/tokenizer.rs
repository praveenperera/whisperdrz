//! Tokenizer encode/decode against python's tiktoken, from `tests/data/tokenizer.json`
//! (written by `rust/tools/dump_decode_reference.py`): edge cases, seeded random text,
//! random token sequences, and every text and `cur` the reference traces hold

use std::collections::HashSet;

use eyre::{Result, bail, ensure, eyre};
use serde_json::Value;
use sha2::{Digest, Sha256};
use whisperdrz_fixtures::{TraceSet, root};
use whisperdrz_native::text::Tokenizer;

fn load() -> Result<Value> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/tokenizer.json");
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

fn ids(v: &Value) -> Result<Vec<u32>> {
    v.as_array()
        .ok_or_else(|| eyre!("not a list"))?
        .iter()
        .map(|x| {
            x.as_u64()
                .map(|x| x as u32)
                .ok_or_else(|| eyre!("bad id {x}"))
        })
        .collect()
}

#[test]
fn encode_and_decode_match_python() -> Result<()> {
    let want = load()?;
    let tok = Tokenizer::multilingual()?;

    let cases = want["encode"].as_array().ok_or_else(|| eyre!("encode"))?;
    for case in cases {
        let text = case[0].as_str().ok_or_else(|| eyre!("text"))?;
        let got = tok.encode(text)?;
        ensure!(
            got == ids(&case[1])?,
            "encode({text:?}): python {} rust {got:?}",
            case[1]
        );
        ensure!(
            tok.decode(&got)? == text,
            "decode(encode({text:?})) does not round-trip"
        );
    }

    let sequences = want["decode"].as_array().ok_or_else(|| eyre!("decode"))?;
    for case in sequences {
        let seq = ids(&case[0])?;
        let got = tok.decode(&seq)?;
        ensure!(
            Some(got.as_str()) == case[1].as_str(),
            "decode({seq:?}): python {} rust {got:?}",
            case[1]
        );
    }

    eprintln!(
        "{} encode and {} decode cases match",
        cases.len(),
        sequences.len()
    );
    Ok(())
}

/// Final texts and every distinct `cur` in the traces, in the generator's order
fn trace_texts() -> Result<Vec<String>> {
    let mut names: Vec<String> = std::fs::read_dir(root())?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| root().join(n).join("summary.json").exists())
        .collect();
    names.sort();

    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let mut push = |s: &str| {
        if seen.insert(s.to_owned()) {
            out.push(s.to_owned());
        }
    };
    for name in names {
        let set = TraceSet::open(&name).ok_or_else(|| eyre!("trace set {name} vanished"))?;
        for stem in set.stems()? {
            push(
                set.result(&stem)?["text"]
                    .as_str()
                    .ok_or_else(|| eyre!("result without text"))?,
            );
            for ev in set.calls(&stem, "update_state_after_decode")? {
                let cur = ev.after_arg("state")?["__state"]["cur"]
                    .as_str()
                    .ok_or_else(|| eyre!("state without cur"))?;
                push(cur);
            }
        }
    }

    Ok(out)
}

#[test]
fn encode_matches_python_on_trace_texts() -> Result<()> {
    if TraceSet::open("short3").is_none() {
        return Ok(());
    }

    let want = &load()?["trace_texts"];
    let texts = trace_texts()?;
    let joined = format!("{:x}", Sha256::digest(texts.join("\0").as_bytes()));
    ensure!(
        want["count"].as_u64() == Some(texts.len() as u64)
            && want["sha256"].as_str() == Some(&joined),
        "trace texts differ from the ones the reference was generated from (python {}, rust {}); regenerate tests/data",
        want["count"],
        texts.len()
    );

    let digests = want["ids_sha256"]
        .as_array()
        .ok_or_else(|| eyre!("ids_sha256"))?;
    let tok = Tokenizer::multilingual()?;
    for (text, d) in texts.iter().zip(digests) {
        let got = tok.encode(text)?;
        let joined = got.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
        let digest = format!("{:x}", Sha256::digest(joined.as_bytes()));
        if Some(&digest[..8]) != d.as_str() {
            bail!("encode({text:?}) differs from python: rust {got:?}");
        }
    }

    eprintln!("{} trace texts encode as in python", texts.len());
    Ok(())
}
