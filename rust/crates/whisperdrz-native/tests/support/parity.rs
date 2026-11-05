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

    /// Ours over the max-abs bar
    pub fn max_ratio(&self) -> f64 {
        self.ours.max_abs / self.max_bar.max(f64::MIN_POSITIVE)
    }

    /// Ours over torch FP16's mean-abs
    pub fn mean_ratio(&self) -> f64 {
        self.ours.mean_abs / self.torch.mean_abs.max(f64::MIN_POSITIVE)
    }

    /// This row as a miss named `name`, judged on `check`'s bars
    pub fn miss(&self, name: String, check: Check) -> Miss {
        Miss {
            name,
            max_ratio: (check == Check::Full).then(|| self.max_ratio()),
            mean_ratio: self.mean_ratio(),
        }
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

/// A row past its bar, with ours over each bar its check asserts
pub struct Miss {
    pub name: String,
    /// `None` where only the mean-abs bar is asserted
    pub max_ratio: Option<f64>,
    pub mean_ratio: f64,
}

/// Prints a table and returns the names of the rows that fail `check`; failing rows are
/// marked, and a [`Check::Off`] table marks and returns none
pub fn report(title: &str, rows: &[Row], check: Check) -> Vec<String> {
    report_misses(title, rows, check)
        .into_iter()
        .map(|m| m.name)
        .collect()
}

/// [`report`], returning each failing row as a [`Miss`] with its ratios
pub fn report_misses(title: &str, rows: &[Row], check: Check) -> Vec<Miss> {
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
            r.max_ratio(),
            r.ours.mean_abs,
            r.torch.mean_abs,
            r.mean_ratio(),
            if r.passes_check(check) { "" } else { "  FAIL" }
        );
    }
    rows.iter()
        .filter(|r| !r.passes_check(check))
        .map(|r| r.miss(format!("{title}: {}", r.name), check))
        .collect()
}
