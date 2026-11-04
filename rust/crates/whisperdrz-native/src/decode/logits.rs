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

use super::mask::TokenMask;
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

/// A logits row under an additive mask, `x + (blocked ? -65504 : unblocked)` rounded to
/// fp16 as [`masked_row`] computes it, with the argmax taken without building the row
///
/// The scan covers unblocked entries outside `skip`, where the masked value is the logit
/// itself (adding a signed zero changes at most the sign of a zero, which compares
/// equal), and keeps only the best and runner-up. [`Self::argmax`] then merges in the
/// caller's values for the `skip` entries, which is exact because the top two of a
/// union are the top two of the parts' top twos under argmax's order (value, then
/// lowest id). Blocked entries sit near -65504 and only compete when the row has a
/// NaN, fewer than two other entries, or a logit large enough to stay above the
/// runner-up after the fill; those rows take the full [`masked_row`] path
pub struct MaskedRow<'a> {
    values: &'a [f16],
    blocked: &'a TokenMask,
    unblocked: f32,
    skip: &'a [Token],
    top: Top2,
    /// No row entry is NaN
    finite: bool,
    /// An upper bound on every blocked entry's masked value
    blocked_bound: f32,
}

/// Best and runner-up `(id, value)` under [`argmax_wide`]'s order
#[derive(Clone, Copy, Default)]
struct Top2 {
    best: Option<(usize, f32)>,
    second: Option<(usize, f32)>,
}

impl Top2 {
    /// Adds an entry with a higher id than every entry so far, as [`argmax_wide`]'s loop
    fn push(&mut self, i: usize, v: f32) {
        let rank = |v: f32| if v.is_nan() { f32::INFINITY } else { v };
        let better =
            |a: f32, b: f32| (a.is_nan() && !b.is_nan()) || (!b.is_nan() && rank(a) > rank(b));
        match self.best {
            None => self.best = Some((i, v)),
            Some((_, bv)) if better(v, bv) => {
                self.second = self.best;
                self.best = Some((i, v));
            }
            Some(_) => {
                if self.second.is_none_or(|(_, sv)| better(v, sv)) {
                    self.second = Some((i, v));
                }
            }
        }
    }
}

/// Entries per scan chunk, one mask word
const CHUNK: usize = 64;
/// Independent accumulators per chunk reduction, so the max reductions vectorize
const LANES: usize = 8;

impl<'a> MaskedRow<'a> {
    /// Scans `values` under `blocked` once. `skip` lists the ids [`Self::argmax`] gives
    /// values for, sorted and unique
    pub fn new(
        values: &'a [f16],
        blocked: &'a TokenMask,
        unblocked: f32,
        skip: &'a [Token],
    ) -> Self {
        assert_eq!(
            values.len(),
            blocked.len(),
            "logits and mask lengths differ"
        );
        assert!(
            skip.windows(2).all(|w| w[0] < w[1]),
            "skip ids must be sorted and unique"
        );

        let scan = scan(values, blocked, skip);
        let (top, raw_max, nan) = (scan.top, scan.raw_max, scan.nan);
        let blocked_bound = f16::from_f32(raw_max + MASK_FILL).to_f32();
        Self {
            values,
            blocked,
            unblocked,
            skip,
            top,
            finite: !nan,
            blocked_bound,
        }
    }

    /// Entry `id` of the masked row
    pub fn value(&self, id: Token) -> f16 {
        let x = self.values[id as usize].to_f32();
        let add = if self.blocked.get(id) {
            MASK_FILL
        } else {
            self.unblocked
        };
        f16::from_f32(x + add)
    }

    /// [`argmax_wide`] of the masked row with entry `skip[k]` replaced by `at_skip[k]`
    pub fn argmax(&self, at_skip: &[f16]) -> Argmax {
        assert_eq!(at_skip.len(), self.skip.len(), "one value per skipped id");

        let mut cands: Vec<(usize, f32)> = [self.top.best, self.top.second]
            .into_iter()
            .flatten()
            .chain(
                self.skip
                    .iter()
                    .zip(at_skip)
                    .map(|(&t, v)| (t as usize, v.to_f32())),
            )
            .collect();
        cands.sort_unstable_by_key(|&(i, _)| i);
        let mut top = Top2::default();
        for &(i, v) in &cands {
            top.push(i, v);
        }

        let fast = match (top.best, top.second) {
            (Some(best), Some(second))
                if self.finite
                    && at_skip.iter().all(|v| !v.is_nan())
                    && self.blocked_bound < second.1 =>
            {
                Some((best.0, second.0))
            }
            _ => None,
        };
        let Some((best, second)) = fast else {
            return self.argmax_full(at_skip);
        };

        let value_at = |i: usize| match self.skip.binary_search(&(i as Token)) {
            Ok(k) => at_skip[k],
            Err(_) => self.value(i as Token),
        };
        Argmax {
            token: best as Token,
            value: value_at(best),
            runner_up: Some((second as Token, value_at(second))),
        }
    }

    /// The row built in full, for the rows the scan can't decide
    fn argmax_full(&self, at_skip: &[f16]) -> Argmax {
        let wide = widen(self.values);
        let (mut row, mut row_wide) = masked_row(&wide, |i| self.blocked.get(i), self.unblocked);
        for (&t, &v) in self.skip.iter().zip(at_skip) {
            row[t as usize] = v;
            row_wide[t as usize] = v.to_f32();
        }

        argmax_wide(&row, &row_wide)
    }
}

/// What [`MaskedRow::new`]'s scan finds: the eligible top two, the raw max over every
/// entry and whether any entry is NaN
struct Scan {
    top: Top2,
    raw_max: f32,
    nan: bool,
}

/// The scan with the chunk statistics on AVX2 and F16C where the CPU has them. Both
/// paths compute the same raw max and NaN flag, so they pick the same chunks
fn scan(values: &[f16], blocked: &TokenMask, skip: &[Token]) -> Scan {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("f16c") {
        // SAFETY: the CPU supports both features, checked just above
        return unsafe { scan_avx2(values, blocked, skip) };
    }

    scan_with(values, blocked, skip, chunk_stats)
}

/// [`scan_with`] compiled for AVX2 and F16C, so [`chunk_stats_avx2`] inlines into it
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn scan_avx2(values: &[f16], blocked: &TokenMask, skip: &[Token]) -> Scan {
    scan_with(values, blocked, skip, |chunk| {
        // SAFETY: this function only runs where both features are present
        unsafe { chunk_stats_avx2(chunk) }
    })
}

/// The raw max and NaN flag of one scan chunk, branch-free so the max reduction
/// vectorizes (a compare-select is `maxps`; NaN is tracked apart)
fn chunk_stats(chunk: &[f16]) -> (f32, bool) {
    let mut wide = [0f32; CHUNK];
    let wide = &mut wide[..chunk.len()];
    chunk.convert_to_f32_slice(wide);

    let mut raw = [f32::NEG_INFINITY; LANES];
    let mut any_nan = [false; LANES];
    let (lanes, tail) = wide.as_chunks::<LANES>();
    for c in lanes {
        for l in 0..LANES {
            let v = c[l];
            raw[l] = if v > raw[l] { v } else { raw[l] };
            any_nan[l] |= v.is_nan();
        }
    }

    for (l, &v) in tail.iter().enumerate() {
        raw[l] = if v > raw[l] { v } else { raw[l] };
        any_nan[l] |= v.is_nan();
    }

    let max = raw.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    (max, any_nan.iter().any(|&b| b))
}

/// [`chunk_stats`] on a full chunk with `vcvtph2ps` and `vmaxps`, without storing the
/// widened values. `max_ps(v, acc)` is `v > acc ? v : acc`, the scalar select, so a
/// NaN never enters the max; the partial last chunk takes the scalar path
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn chunk_stats_avx2(chunk: &[f16]) -> (f32, bool) {
    use std::arch::x86_64::{
        __m128i, _CMP_UNORD_Q, _mm_loadu_si128, _mm256_cmp_ps, _mm256_cvtph_ps, _mm256_max_ps,
        _mm256_movemask_ps, _mm256_or_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    if chunk.len() != CHUNK {
        return chunk_stats(chunk);
    }

    let p = chunk.as_ptr().cast::<__m128i>();
    let mut acc = [_mm256_set1_ps(f32::NEG_INFINITY); 4];
    let mut nan = _mm256_setzero_ps();
    for k in 0..CHUNK / 8 {
        // SAFETY: the chunk holds CHUNK f16s, so all eight 16-byte loads are in bounds
        let v = _mm256_cvtph_ps(unsafe { _mm_loadu_si128(p.add(k)) });
        acc[k % 4] = _mm256_max_ps(v, acc[k % 4]);
        nan = _mm256_or_ps(nan, _mm256_cmp_ps::<_CMP_UNORD_Q>(v, v));
    }

    let m = _mm256_max_ps(_mm256_max_ps(acc[0], acc[1]), _mm256_max_ps(acc[2], acc[3]));
    let mut lanes = [0f32; 8];
    // SAFETY: `lanes` holds eight f32s
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), m) };
    let max = lanes.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    (max, _mm256_movemask_ps(nan) != 0)
}

/// The scan behind [`MaskedRow::new`], with `stats` giving each chunk's raw max and NaN
/// flag. The raw max bounds the eligible max, so a chunk at or below the runner-up is
/// skipped without widening it or walking its bits
#[inline(always)]
fn scan_with(
    values: &[f16],
    blocked: &TokenMask,
    skip: &[Token],
    stats: impl Fn(&[f16]) -> (f32, bool),
) -> Scan {
    let mut top = Top2::default();
    let mut raw_max = f32::NEG_INFINITY;
    let mut nan = false;
    let mut wide = [0f32; CHUNK];
    let mut skip_at = 0;
    for (c, (chunk, &word)) in values.chunks(CHUNK).zip(blocked.words()).enumerate() {
        let base = c * CHUNK;
        let mut eligible = !word;
        while skip_at < skip.len() && (skip[skip_at] as usize) < base + CHUNK {
            eligible &= !(1u64 << (skip[skip_at] as usize - base));
            skip_at += 1;
        }

        let (chunk_max, chunk_nan) = stats(chunk);
        raw_max = raw_max.max(chunk_max);
        nan |= chunk_nan;

        // later ids lose ties, so a chunk can only matter if it beats the runner-up
        if eligible == 0 || top.second.is_some_and(|(_, sv)| chunk_max <= sv) {
            continue;
        }

        let wide = &mut wide[..chunk.len()];
        chunk.convert_to_f32_slice(wide);
        let mut bits = eligible & (u64::MAX >> (CHUNK - wide.len()));
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            top.push(base + j, wide[j]);
        }
    }

    Scan { top, raw_max, nan }
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
    fn masked_row_argmax_matches_full_row() {
        use super::{MaskedRow, argmax_wide, masked_row, widen};
        use crate::decode::mask::TokenMask;
        use crate::text::Token;

        // xorshift, so the cases are fixed
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        // a few repeated values for ties, signed zeros, fp16 extremes and any bit pattern
        let draw = |r: u64| -> f16 {
            match r % 8 {
                0..=2 => f16::from_f32([1.0, 2.0, -3.0, 0.5][(r >> 8) as usize % 4]),
                3 => {
                    f16::from_bits([0x0000, 0x8000, 0x7c00, 0xfc00, 0x7bff][(r >> 8) as usize % 5])
                }
                4 => f16::from_bits((r >> 16) as u16),
                _ => f16::from_f32(((r >> 16) % 2000) as f32 / 100.0 - 10.0),
            }
        };

        let same = |a: f16, b: f16| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
        for case in 0..20_000 {
            let len = 1 + (next() % 200) as usize;
            let values: Vec<f16> = (0..len).map(|_| draw(next())).collect();
            let density = [0, 50, 95, 99, 100][case % 5];
            let blocked = TokenMask::from_ids(
                len,
                (0..len as Token).filter(|_| ((next() % 100) as usize) < density),
            );
            let mut skip: Vec<Token> = (0..next() % 6)
                .map(|_| (next() % len as u64) as Token)
                .collect();
            skip.sort_unstable();
            skip.dedup();
            let at_skip: Vec<f16> = skip.iter().map(|_| draw(next())).collect();
            let unblocked = if case % 2 == 0 { 0.0 } else { -0.0 };

            let (mut row, mut row_wide) =
                masked_row(&widen(&values), |i| blocked.get(i), unblocked);
            for (&t, &v) in skip.iter().zip(&at_skip) {
                row[t as usize] = v;
                row_wide[t as usize] = v.to_f32();
            }
            let want = argmax_wide(&row, &row_wide);

            let masked = MaskedRow::new(&values, &blocked, unblocked, &skip);
            for (i, &r) in row.iter().enumerate() {
                if skip.binary_search(&(i as Token)).is_err() {
                    assert!(same(masked.value(i as Token), r), "case {case} value {i}");
                }
            }
            let got = masked.argmax(&at_skip);
            assert_eq!(got.token, want.token, "case {case}");
            assert!(same(got.value, want.value), "case {case}");
            assert_eq!(
                got.runner_up.map(|r| r.0),
                want.runner_up.map(|r| r.0),
                "case {case}"
            );
            if let (Some(g), Some(w)) = (got.runner_up, want.runner_up) {
                assert!(same(g.1, w.1), "case {case}");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_chunk_stats_match_scalar() {
        use super::{CHUNK, chunk_stats, chunk_stats_avx2};

        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("f16c"))
        {
            eprintln!("skipping: no AVX2 and F16C");
            return;
        }

        // every fp16 bit pattern in order, then strided so NaNs, infinities and signed
        // zeros share chunks with ordinary values
        let ordered: Vec<f16> = (0..=u16::MAX).map(f16::from_bits).collect();
        let strided: Vec<f16> = (0..=u16::MAX)
            .map(|i| f16::from_bits(i.wrapping_mul(40_503)))
            .collect();
        let mut nans = 0;
        for values in [ordered, strided] {
            for chunk in values.chunks(CHUNK) {
                let want = chunk_stats(chunk);
                // SAFETY: both features were detected above
                let got = unsafe { chunk_stats_avx2(chunk) };
                assert_eq!(got.0.to_bits(), want.0.to_bits(), "{chunk:?}");
                assert_eq!(got.1, want.1, "{chunk:?}");
                nans += usize::from(want.1);
            }
        }

        assert!(nans > 0);
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
