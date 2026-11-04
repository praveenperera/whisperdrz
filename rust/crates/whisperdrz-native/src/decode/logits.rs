//! The float operations on the decision path, reproduced bit for bit
//!
//! Python runs them on CUDA fp16 tensors. Every elementwise op there computes in fp32
//! (`opmath_t`) and rounds once to fp16, which the CPU does the same way:
//!
//! - `logits + additive`, the additive mask being `0` or `-65504` (fp16 max), see
//!   [`add_mask`]
//! - the repetition penalty `where(x > 0, x / 1.2, x * 1.2)`. Dividing by a CPU scalar
//!   on CUDA is `x * (1.0f / 1.2f)` (`BinaryDivTrueKernel.cu`, torch 2.8), not a true
//!   division, see [`penalize`]
//! - `-inf` fills and `argmax`, which takes the first index among equal maxima and
//!   treats NaN as the maximum
//!
//! Temperature is 0 (greedy), so no softmax or sampling is involved

use half::f16;
use half::slice::HalfFloatSliceExt;

use crate::text::Token;

/// The additive mask value: fp16 max, not `-inf`, so a fully blocked row stays finite
pub const MASK_FILL: f32 = -65504.0;
/// `REP_PENALTY` in `decode_batch`
pub const REP_PENALTY: f32 = 1.2;

/// `x + (blocked ? -65504 : 0)` in fp16
pub fn add_mask(x: f16, blocked: bool) -> f16 {
    f16::from_f32(x.to_f32() + if blocked { MASK_FILL } else { 0.0 })
}

/// The Phase 4 resample mask, `x + bool.to(fp16) * -65504`, where an unblocked entry
/// adds `-0.0`
pub fn add_resample_mask(x: f16, blocked: bool) -> f16 {
    f16::from_f32(x.to_f32() + if blocked { MASK_FILL } else { -0.0 })
}

/// The repetition penalty as CUDA computes it: `x * fl(1 / 1.2)` for positive `x`,
/// `x * 1.2` otherwise
pub fn penalize(x: f16) -> f16 {
    let v = x.to_f32();
    if v > 0.0 {
        f16::from_f32(v * (1.0f32 / REP_PENALTY))
    } else {
        f16::from_f32(v * REP_PENALTY)
    }
}

/// The penalty with a true division, as torch computes it on the CPU. Only used to
/// report how sensitive a decision is to the reciprocal trick
pub fn penalize_true_division(x: f16) -> f16 {
    let v = x.to_f32();
    if v > 0.0 {
        f16::from_f32(v / REP_PENALTY)
    } else {
        f16::from_f32(v * REP_PENALTY)
    }
}

/// The winner of an argmax and its closest competitor
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Argmax {
    pub token: Token,
    pub value: f16,
    /// The best other entry (lowest id among equals), `None` for a one-entry row
    pub runner_up: Option<(Token, f16)>,
}

impl Argmax {
    /// Winner minus runner-up in fp32; 0 means the tie went to the lower id
    pub fn margin(&self) -> Option<f32> {
        self.runner_up
            .map(|(_, v)| self.value.to_f32() - v.to_f32())
    }
}

/// `torch.argmax`: first index of the maximum, NaN counting as the maximum
pub fn argmax(values: &[f16]) -> Argmax {
    argmax_wide(values, &widen(values))
}

/// `values` widened to f32, exactly, with half's vectorized conversion
pub fn widen(values: &[f16]) -> Vec<f32> {
    let mut wide = vec![0f32; values.len()];
    values.convert_to_f32_slice(&mut wide);
    wide
}

/// [`argmax`] over `values` given `wide`, their f32 widening. Widening is exact, so
/// comparing the f32s decides exactly as comparing the fp16s; it only skips a scalar
/// conversion per comparison, which dominated the per-row host time
pub fn argmax_wide(values: &[f16], wide: &[f32]) -> Argmax {
    // python never calls this on an empty row
    assert!(!values.is_empty(), "argmax of an empty row");
    assert_eq!(values.len(), wide.len(), "argmax over mismatched widening");

    let rank = |v: f32| if v.is_nan() { f32::INFINITY } else { v };
    let better = |a: f32, b: f32| (a.is_nan() && !b.is_nan()) || (!b.is_nan() && rank(a) > rank(b));

    let mut best = 0usize;
    let mut second: Option<usize> = None;
    for i in 1..wide.len() {
        let v = wide[i];
        if better(v, wide[best]) {
            second = Some(best);
            best = i;
        } else if second.is_none_or(|s| better(v, wide[s])) {
            second = Some(i);
        }
    }

    Argmax {
        token: best as Token,
        value: values[best],
        runner_up: second.map(|s| (s as Token, values[s])),
    }
}

/// A logits row plus an additive mask, `x + (blocked(i) ? -65504 : unblocked)` in fp32
/// rounded once to fp16, as [`add_mask`] (`unblocked` 0) and [`add_resample_mask`]
/// (`unblocked` -0) compute it per entry. Returns the fp16 row and its exact widening
pub fn masked_row(
    wide: &[f32],
    blocked: impl Fn(u32) -> bool,
    unblocked: f32,
) -> (Vec<f16>, Vec<f32>) {
    let mut sum: Vec<f32> = wide
        .iter()
        .zip(0..)
        .map(|(&x, i)| x + if blocked(i) { MASK_FILL } else { unblocked })
        .collect();
    let mut row = vec![f16::ZERO; wide.len()];
    // the slice conversions round to nearest even like `f16::from_f32`
    row.convert_from_f32_slice(&sum);
    row.convert_to_f32_slice(&mut sum);
    (row, sum)
}

#[cfg(test)]
mod tests {
    use half::f16;

    use super::{add_mask, argmax, penalize};

    #[test]
    fn mask_rounding_at_fp16_max() {
        // spacing near 65504 is 32: small logits round onto the fill value, large
        // positive ones stay above it, and anything below -16 overflows to -inf
        assert_eq!(add_mask(f16::from_f32(10.0), true).to_f32(), -65504.0);
        assert_eq!(add_mask(f16::from_f32(20.0), true).to_f32(), -65472.0);
        assert!(add_mask(f16::from_f32(-17.0), true).to_f32().is_infinite());
    }

    #[test]
    fn penalty_reciprocal_differs_from_division() {
        // an fp16 value where x * fl(1/1.2) and x / 1.2 round to different fp16s, so the
        // emulation has to follow the CUDA kernel rather than the math
        let differs = (0..=0x7bffu16)
            .map(f16::from_bits)
            .filter(|x| x.to_f32() > 0.0)
            .any(|x| penalize(x) != f16::from_f32(x.to_f32() / 1.2));
        assert!(differs);
    }

    #[test]
    fn masked_row_matches_per_entry_ops() {
        // every fp16 bit pattern, blocked and not, under both unblocked addends, so the
        // vectorized row path must round exactly as the scalar ops do (NaNs compared as NaN)
        let logits: Vec<f16> = (0..=u16::MAX).map(f16::from_bits).collect();
        let wide = super::widen(&logits);
        for blocked_every in [1, 2, 3] {
            let blocked = |i: u32| i.is_multiple_of(blocked_every);
            for (unblocked, scalar) in [
                (0.0, add_mask as fn(f16, bool) -> f16),
                (-0.0, super::add_resample_mask),
            ] {
                let (row, row_wide) = super::masked_row(&wide, blocked, unblocked);
                for (i, (&x, (&r, &w))) in logits.iter().zip(row.iter().zip(&row_wide)).enumerate()
                {
                    let want = scalar(x, blocked(i as u32));
                    assert!(
                        r.to_bits() == want.to_bits() || (r.is_nan() && want.is_nan()),
                        "entry {i}"
                    );
                    assert!(w.to_bits() == r.to_f32().to_bits() || (w.is_nan() && r.is_nan()));
                }
            }
        }
    }

    #[test]
    fn argmax_takes_first_max() {
        let row: Vec<f16> = [1.0, 3.0, 2.0, 3.0].map(f16::from_f32).to_vec();
        let a = argmax(&row);
        assert_eq!(a.token, 1);
        assert_eq!(a.runner_up.map(|r| r.0), Some(3));
        assert_eq!(a.margin(), Some(0.0));
    }
}
