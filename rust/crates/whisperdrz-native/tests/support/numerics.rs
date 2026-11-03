//! Shared helpers of the GPU numerics tests: random FP16 data, FP16 spacing and error
//! statistics that fail on non-finite values and length mismatches

use eyre::{Result, ensure};
use half::f16;

/// FP16 bits of a quiet NaN, for prefilling outputs a kernel must overwrite
pub const NAN16: u16 = 0x7e00;

/// SplitMix64 with Box-Muller normals, so tests need no RNG crate and stay reproducible
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1)
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    pub fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }

    /// `n` FP16 bit patterns of `mean + std * N(0, 1)`
    pub fn f16_normal(&mut self, n: usize, mean: f64, std: f64) -> Vec<u16> {
        (0..n)
            .map(|_| f16::from_f64(mean + std * self.normal()).to_bits())
            .collect()
    }
}

pub fn f16_to_f64(bits: u16) -> f64 {
    f16::from_bits(bits).to_f64()
}

pub fn f16_round(x: f64) -> f64 {
    f16::from_f64(x).to_f64()
}

/// The spacing of FP16 values around `x`: one unit in the last place
pub fn ulp16(x: f64) -> f64 {
    let a = x.abs();
    if a < 2f64.powi(-14) {
        2f64.powi(-24)
    } else {
        2f64.powi(a.log2().floor() as i32 - 10)
    }
}

/// Max and mean absolute error of `got` against `want`
#[derive(Clone, Copy, Debug, Default)]
pub struct Err {
    pub max_abs: f64,
    pub mean_abs: f64,
}

impl Err {
    /// Fails on a length mismatch, an empty comparison or any non-finite value on either
    /// side, so a kernel that wrote nothing (NaN prefill) or too little can't pass
    pub fn of(got: &[f64], want: &[f64]) -> Result<Self> {
        ensure!(
            got.len() == want.len() && !got.is_empty(),
            "comparing {} values against {}",
            got.len(),
            want.len()
        );
        let (mut max_abs, mut sum) = (0f64, 0f64);
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            ensure!(
                g.is_finite() && w.is_finite(),
                "non-finite value at {i}: got {g}, want {w}"
            );
            let d = (g - w).abs();
            max_abs = max_abs.max(d);
            sum += d;
        }

        Ok(Self {
            max_abs,
            mean_abs: sum / got.len() as f64,
        })
    }
}

/// Whether the GPU tests that need nothing but a GPU should run: `rust/scripts/gpu-test.sh`
/// sets `WHISPERDRZ_GPU=1` inside the GPU queue job
pub fn gpu_enabled() -> bool {
    let on = std::env::var_os("WHISPERDRZ_GPU").is_some_and(|v| v == "1");
    if !on {
        eprintln!("skipping: WHISPERDRZ_GPU not set (run scripts/gpu-test.sh)");
    }
    on
}
