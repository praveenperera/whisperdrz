//! Parity tables of the decoder GPU tests: ours and torch FP16 judged against a common
//! FP32 truth, plus `.npy` and device readers

use std::path::Path;

use eyre::{Result, bail, ensure, eyre};
use serde_json::Value;
use whisperdrz_fixtures::{Array, ArrayData, TraceSet, read_npy};
use whisperdrz_native::cuda::{Buf, Gpu};

use crate::numerics::{Err, f16_to_f64};

/// Ours may be this much worse than torch FP16 against the FP32 truth
pub const MAX_RATIO: f64 = 1.5;

/// An FP16 `.npy` as bits, with its shape
pub fn f16_bits(path: &Path) -> Result<(Vec<usize>, Vec<u16>)> {
    let a = read_npy(path)?;
    let ArrayData::F16(v) = a.data else {
        bail!("{} is not float16", path.display())
    };
    Ok((a.shape, v.iter().map(|h| h.to_bits()).collect()))
}

/// An array argument or result of a trace event in its recorded shape. Blobs are stored by
/// content, so a blob first saved by another call with the same bytes (a slot's `[1500,
/// 1280]` encoder output under a `[1, 1500, 1280]` batch) carries that call's shape, which
/// `TraceSet::array` refuses; this takes the recorded shape when the element counts agree
pub fn trace_array(set: &TraceSet, v: &Value) -> Result<Array> {
    let Some(blob) = v.get("blob").and_then(Value::as_str) else {
        return set.array(v);
    };
    let shape = v["shape"]
        .as_array()
        .ok_or_else(|| eyre!("array without shape"))?
        .iter()
        .map(|d| {
            d.as_u64()
                .map(|d| d as usize)
                .ok_or_else(|| eyre!("bad dim {d}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut a = read_npy(&set.blob_path(blob))?;
    ensure!(
        a.shape.iter().product::<usize>() == shape.iter().product::<usize>(),
        "blob {blob} shape {:?} holds a different count than recorded {shape:?}",
        a.shape
    );
    a.shape = shape;
    Ok(a)
}

/// Any float `.npy` widened to f64
pub fn f64s(path: &Path) -> Result<Vec<f64>> {
    Ok(read_npy(path)?
        .to_f32()?
        .into_iter()
        .map(f64::from)
        .collect())
}

pub fn bits_f64(bits: &[u16]) -> Vec<f64> {
    bits.iter().copied().map(f16_to_f64).collect()
}

/// The first `n` FP16 values of `buf`
pub fn download16(gpu: &Gpu, buf: &Buf, n: usize) -> Result<Vec<u16>> {
    let mut host = vec![0u16; n];
    gpu.download_into(buf, 0, &mut host)?;
    Ok(host)
}

/// One comparison row: ours and torch FP16 against the truth
pub struct Row {
    pub name: String,
    pub ours: Err,
    pub torch: Err,
    /// The max-abs error ours is judged against: torch FP16's, or for chained layers the
    /// worst valid torch FP16 variant's
    pub max_bar: f64,
}

impl Row {
    pub fn new(name: &str, ours: &[f64], torch: &[f64], truth: &[f64]) -> Result<Self> {
        let torch = Err::of(torch, truth)?;
        Ok(Self {
            name: name.to_owned(),
            ours: Err::of(ours, truth)?,
            max_bar: torch.max_abs,
            torch,
        })
    }

    /// Raises the max-abs bar to the worst of these variant rows' own errors
    pub fn bar_from(mut self, variants: &[&Row]) -> Self {
        for v in variants {
            self.max_bar = self.max_bar.max(v.ours.max_abs);
        }
        self
    }

    /// Exact agreement on both sides counts as a pass, as 0 <= 1.5 * 0
    pub fn passes(&self) -> bool {
        self.ours.max_abs <= MAX_RATIO * self.max_bar
            && self.ours.mean_abs <= MAX_RATIO * self.torch.mean_abs
    }
}

/// Prints a table and returns the names of the rows that fail; a table that is not
/// `checked` prints no failure marks and returns none
pub fn report(title: &str, rows: &[Row], checked: bool) -> Vec<String> {
    println!("\n{title}");
    println!(
        "{:<12} {:>11} {:>11} {:>11} {:>6} {:>11} {:>11} {:>6}",
        "layer", "ours max", "torch max", "max bar", "ratio", "ours mean", "torch mean", "ratio"
    );
    for r in rows {
        println!(
            "{:<12} {:>11.4e} {:>11.4e} {:>11.4e} {:>6.2} {:>11.4e} {:>11.4e} {:>6.2}{}",
            r.name,
            r.ours.max_abs,
            r.torch.max_abs,
            r.max_bar,
            r.ours.max_abs / r.max_bar,
            r.ours.mean_abs,
            r.torch.mean_abs,
            r.ours.mean_abs / r.torch.mean_abs,
            if !checked || r.passes() { "" } else { "  FAIL" }
        );
    }
    rows.iter()
        .filter(|r| checked && !r.passes())
        .map(|r| format!("{title}: {}", r.name))
        .collect()
}
