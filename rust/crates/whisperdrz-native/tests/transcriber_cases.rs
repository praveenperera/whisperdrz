//! The scheduler's CPython-dependent pieces against values recorded from Python
//! (`rust/tools/gen_transcriber_cases.py`): `set` iteration order after random
//! add/discard histories, and the decode-step bucket choice

use eyre::{Result, ensure, eyre};
use serde_json::Value;
use whisperdrz_native::transcriber::{PySet, captured_buckets, get_bucket};

fn data() -> Result<Value> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/transcriber_cases.json"
    );
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

fn usizes(v: &Value) -> Result<Vec<usize>> {
    v.as_array()
        .ok_or_else(|| eyre!("not a list: {v}"))?
        .iter()
        .map(|x| {
            x.as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| eyre!("not an int: {x}"))
        })
        .collect()
}

#[test]
fn set_order_follows_cpython() -> Result<()> {
    let data = data()?;
    let cases = data["sets"].as_array().ok_or_else(|| eyre!("no sets"))?;
    let mut ops = 0;
    let mut unordered = 0;
    for (ci, case) in cases.iter().enumerate() {
        let mut set = PySet::from_items(usizes(&case["initial"])?);
        ensure!(
            set.to_vec() == usizes(&case["start"])?,
            "case {ci}: start order"
        );

        for (oi, op) in case["ops"]
            .as_array()
            .ok_or_else(|| eyre!("no ops"))?
            .iter()
            .enumerate()
        {
            let key = op[1].as_u64().ok_or_else(|| eyre!("bad key"))? as usize;
            match op[0].as_str() {
                Some("add") => set.add(key),
                Some("discard") => set.discard(key),
                other => eyre::bail!("unknown op {other:?}"),
            }

            let want = usizes(&op[2])?;
            ensure!(
                set.to_vec() == want,
                "case {ci} op {oi} ({op}): python {want:?} rust {:?}",
                set.to_vec()
            );
            ensure!(set.len() == want.len() && set.contains(key) == want.contains(&key));
            ops += 1;
            unordered += usize::from(!want.is_sorted());
        }
    }

    // the data has to exercise non-ascending orders, or this checks nothing
    ensure!(
        unordered > 100,
        "only {unordered} of {ops} states are out of order"
    );
    Ok(())
}

#[test]
fn buckets_follow_capture_all() -> Result<()> {
    let data = data()?;
    let cases = data["buckets"]
        .as_array()
        .ok_or_else(|| eyre!("no buckets"))?;
    ensure!(!cases.is_empty());
    for case in cases {
        let max_bucket = case["max_bucket"]
            .as_u64()
            .ok_or_else(|| eyre!("max_bucket"))? as usize;
        let buckets = captured_buckets(max_bucket);
        ensure!(
            buckets == usizes(&case["buckets"])?,
            "max_bucket {max_bucket}: {buckets:?}"
        );
        for (batch, want) in usizes(&case["choice"])?.into_iter().enumerate() {
            ensure!(
                get_bucket(batch, &buckets) == want,
                "max_bucket {max_bucket}, batch {batch}: want {want}"
            );
        }
    }

    Ok(())
}
