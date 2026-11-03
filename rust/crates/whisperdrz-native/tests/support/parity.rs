//! Parity tables of the decoder GPU tests: ours and torch FP16 judged against a common
//! FP32 truth, plus `.npy` and device readers

use std::path::Path;

use eyre::{Result, bail};
use whisperdrz_fixtures::{ArrayData, read_npy};
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
        self.ours.max_abs <= MAX_RATIO * self.max_bar && self.passes_mean()
    }

    pub fn passes_mean(&self) -> bool {
        self.ours.mean_abs <= MAX_RATIO * self.torch.mean_abs
    }

    fn passes_check(&self, check: Check) -> bool {
        match check {
            Check::Full => self.passes(),
            Check::Mean => self.passes_mean(),
            Check::Off => true,
        }
    }
}

/// Which bars a [`report`] table asserts
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Check {
    /// Max-abs and mean-abs
    Full,
    /// Mean-abs only; max-abs is printed for reading
    Mean,
    /// Neither: a yardstick table
    Off,
}

/// Prints a table and returns the names of the rows that fail `check`; failing rows are
/// marked, and a [`Check::Off`] table marks and returns none
pub fn report(title: &str, rows: &[Row], check: Check) -> Vec<String> {
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
            if r.passes_check(check) { "" } else { "  FAIL" }
        );
    }
    rows.iter()
        .filter(|r| !r.passes_check(check))
        .map(|r| format!("{title}: {}", r.name))
        .collect()
}
