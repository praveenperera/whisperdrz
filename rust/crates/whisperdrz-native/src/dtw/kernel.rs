//! DTW path search, word boundaries from the path and anchor bands
//! (`_dtw_numba`, `_dtw_from_neg_matrix`, `_apply_anchor_bands`)

use std::ops::Range;

use eyre::{Result, eyre};

use super::TOKENS_PER_SECOND;
use super::parse::TagAnchor;

/// A row-major f32 cost matrix: text rows by audio columns, lower is better
#[derive(Debug, Clone, PartialEq)]
pub struct NegMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl NegMatrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(eyre!(
                "matrix data has {} values, want {rows}x{cols}",
                data.len()
            ));
        }

        Ok(Self { rows, cols, data })
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    fn fill(&mut self, rows: Range<usize>, cols: Range<usize>, value: f32) {
        for r in rows {
            self.data[r * self.cols..][cols.clone()].fill(value);
        }
    }
}

/// A decoder-emitted timing on chunk word `word` (its index in the chunk's tagged list)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WordAnchor {
    pub word: usize,
    pub start: f64,
    pub end: f64,
}

/// Start and end in seconds of one word group
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WordTime {
    pub start: f64,
    pub end: f64,
}

/// Rows per wavefront strip
const STRIP: usize = 8;

/// Minimum-cost monotone path through `x`, as `_dtw_numba`
///
/// Returns `(text_idx, time_idx)`, both of the path's length. Indices can be -1 where
/// the path runs along the matrix's top or left edge
///
/// Each cell depends on the one above it, so the numba loop is one serial chain per
/// column. Here rows run in strips of [`STRIP`] with lane `k` on row `i0 + k`, `k`
/// columns behind lane 0: the lanes of one step are independent, and each takes its
/// up and diagonal costs from the lane above's last two steps and its left cost from
/// its own last step. Every cell sees the same three costs, comparisons and sum as in
/// the numba loop, so costs and moves are bit for bit the same
pub fn dtw_path(x: &NegMatrix) -> (Vec<i64>, Vec<i64>) {
    #[cfg(target_arch = "x86_64")]
    let strip = strip_sse2;
    #[cfg(not(target_arch = "x86_64"))]
    let strip = strip_scalar;
    dtw_path_with(x, strip)
}

/// One strip's steps: costs of the row above in `top`, the strip's skewed costs in
/// `skew`, its moves into `moves` and its bottom row's costs into `below`
type Strip = fn(skew: &[f32], top: &[f32], below: &mut [f32], moves: &mut [i8], m: usize);

const INF: f32 = f32::INFINITY;

fn dtw_path_with(x: &NegMatrix, strip: Strip) -> (Vec<i64>, Vec<i64>) {
    let (n, m) = (x.rows, x.cols);
    let steps = m + STRIP - 1;
    let strips = n.div_ceil(STRIP);

    // moves per strip, step and lane; lanes outside the matrix write moves never read
    let mut trace = vec![0i8; strips * steps * STRIP];
    // the cost row above the strip, columns 0..=m, padded for lane 0's reads past m
    let mut top = vec![INF; m + 1 + STRIP];
    top[0] = 0.0;
    let mut below = vec![INF; m + 1 + STRIP];
    // the strip's costs skewed to steps: `skew[t][k] = x[i0 + k][t - k]`
    let mut skew = vec![0f32; steps * STRIP];
    for s in 0..strips {
        let i0 = s * STRIP;
        skew.fill(0.0);
        for k in 0..(n - i0).min(STRIP) {
            let row = &x.data[(i0 + k) * m..][..m];
            for (j, &v) in row.iter().enumerate() {
                skew[(j + k) * STRIP + k] = v;
            }
        }

        below.fill(INF);
        let moves = &mut trace[s * steps * STRIP..][..steps * STRIP];
        strip(&skew, &top, &mut below, moves, m);
        std::mem::swap(&mut top, &mut below);
    }

    // row 0 moves left and column 0 moves up
    let move_at = |i: usize, j: usize| -> i8 {
        if i == 0 {
            2
        } else if j == 0 {
            1
        } else {
            let (s, k) = ((i - 1) / STRIP, (i - 1) % STRIP);
            trace[(s * steps + j - 1 + k) * STRIP + k]
        }
    };

    let mut text_idx = Vec::with_capacity(n + m);
    let mut time_idx = Vec::with_capacity(n + m);
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        text_idx.push(i as i64 - 1);
        time_idx.push(j as i64 - 1);
        match move_at(i, j) {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            _ => j -= 1,
        }
    }

    text_idx.reverse();
    time_idx.reverse();
    (text_idx, time_idx)
}

/// The bottom lane's 1-based column at step `t`, when it is inside the matrix
fn bottom_column(t: usize, m: usize) -> Option<usize> {
    (t + 2).checked_sub(STRIP).filter(|&j| j >= 1 && j <= m)
}

/// [`Strip`] lane by lane, for targets without the SSE2 path
#[cfg(any(test, not(target_arch = "x86_64")))]
fn strip_scalar(skew: &[f32], top: &[f32], below: &mut [f32], moves: &mut [i8], m: usize) {
    let mut prev = [INF; STRIP];
    let mut prev2 = [INF; STRIP];
    for t in 0..m + STRIP - 1 {
        let mut up = [0f32; STRIP];
        let mut diag = [0f32; STRIP];
        up[0] = top[t + 1];
        diag[0] = top[t];
        up[1..].copy_from_slice(&prev[..STRIP - 1]);
        diag[1..].copy_from_slice(&prev2[..STRIP - 1]);

        let mut out = [0f32; STRIP];
        for k in 0..STRIP {
            let (c0, c1, c2) = (diag[k], up[k], prev[k]);
            // numba compiles without fastmath, so these are IEEE comparisons (false on
            // NaN)
            let (c, mv) = if c0 <= c1 && c0 <= c2 {
                (c0, 0)
            } else if c1 <= c2 {
                (c1, 1)
            } else {
                (c2, 2)
            };
            out[k] = skew[t * STRIP + k] + c;
            moves[t * STRIP + k] = mv;
        }

        // a lane before its first column holds column 0's cost
        for o in out.iter_mut().skip(t + 1) {
            *o = INF;
        }

        if let Some(j) = bottom_column(t, m) {
            below[j] = out[STRIP - 1];
        }

        prev2 = prev;
        prev = out;
    }
}

/// [`Strip`] with the 8 lanes in two SSE2 registers. The lane shifts stay in
/// registers: the scalar loop's shifted reloads of its last outputs miss store
/// forwarding every step
#[cfg(target_arch = "x86_64")]
fn strip_sse2(skew: &[f32], top: &[f32], below: &mut [f32], moves: &mut [i8], m: usize) {
    use std::arch::x86_64::{
        __m128, __m128i, _mm_add_epi32, _mm_add_ps, _mm_and_ps, _mm_andnot_ps, _mm_andnot_si128,
        _mm_castps_si128, _mm_castsi128_ps, _mm_cmple_ps, _mm_cvtss_f32, _mm_loadu_ps, _mm_move_ss,
        _mm_or_ps, _mm_packs_epi16, _mm_packs_epi32, _mm_set_ss, _mm_set1_epi32, _mm_set1_ps,
        _mm_setzero_si128, _mm_shuffle_ps, _mm_slli_si128, _mm_storel_epi64, _mm_storeu_ps,
    };

    let steps = m + STRIP - 1;
    assert!(skew.len() >= steps * STRIP && moves.len() >= steps * STRIP);
    assert!(top.len() > steps && below.len() > m);

    // SAFETY: SSE2 is part of the x86_64 baseline, and every pointer access is within
    // the lengths asserted above: `skew` and `moves` hold `steps * STRIP` entries
    unsafe {
        // [first, v0, v1, v2]: `v` one lane down, `first` in lane 0
        let shift = |v: __m128, first: __m128| {
            _mm_move_ss(
                _mm_castsi128_ps(_mm_slli_si128::<4>(_mm_castps_si128(v))),
                first,
            )
        };
        let two = _mm_set1_epi32(2);
        // the numba loop's `if c0 <= c1 and c0 <= c2 ... elif c1 <= c2` as masks; `cmpleps`
        // is false on NaN, like numba's IEEE comparisons. Moves are 0, 1 or 2
        let cell = |c0: __m128, c1: __m128, c2: __m128, x: __m128| -> (__m128, __m128i) {
            let take0 = _mm_and_ps(_mm_cmple_ps(c0, c1), _mm_cmple_ps(c0, c2));
            let take1 = _mm_cmple_ps(c1, c2);
            let c12 = _mm_or_ps(_mm_and_ps(take1, c1), _mm_andnot_ps(take1, c2));
            let c = _mm_or_ps(_mm_and_ps(take0, c0), _mm_andnot_ps(take0, c12));
            let mv = _mm_andnot_si128(
                _mm_castps_si128(take0),
                _mm_add_epi32(two, _mm_castps_si128(take1)),
            );
            (_mm_add_ps(x, c), mv)
        };

        let (mut lo, mut hi) = (_mm_set1_ps(INF), _mm_set1_ps(INF));
        let (mut lo2, mut hi2) = (lo, hi);
        for t in 0..steps {
            let up_lo = shift(lo, _mm_set_ss(top[t + 1]));
            let up_hi = shift(hi, _mm_shuffle_ps::<0xff>(lo, lo));
            let diag_lo = shift(lo2, _mm_set_ss(top[t]));
            let diag_hi = shift(hi2, _mm_shuffle_ps::<0xff>(lo2, lo2));
            let x = skew.as_ptr().add(t * STRIP);
            let (mut out_lo, mv_lo) = cell(diag_lo, up_lo, lo, _mm_loadu_ps(x));
            let (mut out_hi, mv_hi) = cell(diag_hi, up_hi, hi, _mm_loadu_ps(x.add(4)));
            let mv = _mm_packs_epi16(_mm_packs_epi32(mv_lo, mv_hi), _mm_setzero_si128());
            _mm_storel_epi64(moves.as_mut_ptr().add(t * STRIP).cast::<__m128i>(), mv);

            // a lane before its first column holds column 0's cost
            if t + 1 < STRIP {
                let mut out = [0f32; STRIP];
                _mm_storeu_ps(out.as_mut_ptr(), out_lo);
                _mm_storeu_ps(out.as_mut_ptr().add(4), out_hi);
                for o in out.iter_mut().skip(t + 1) {
                    *o = INF;
                }

                out_lo = _mm_loadu_ps(out.as_ptr());
                out_hi = _mm_loadu_ps(out.as_ptr().add(4));
            }

            if let Some(j) = bottom_column(t, m) {
                below[j] = _mm_cvtss_f32(_mm_shuffle_ps::<0xff>(out_hi, out_hi));
            }

            (lo2, hi2) = (lo, hi);
            (lo, hi) = (out_lo, out_hi);
        }
    }
}

/// Per-word `(start, end)` from DTW over a prepared (and banded) cost matrix, as
/// `_dtw_from_neg_matrix`
///
/// `word_tokens` are the chunk's token groups, the last one the eot dummy. Entry `i`
/// belongs to word `i`; the result has `min(n_words, word_tokens.len() - 1)` entries,
/// or none when the path is too short for the groups
pub fn dtw_from_neg_matrix(
    neg: &NegMatrix,
    n_words: usize,
    word_tokens: &[Vec<u32>],
) -> Vec<WordTime> {
    if neg.is_empty() || word_tokens.len() <= 1 {
        return Vec::new();
    }

    let (text_idx, time_idx) = dtw_path(neg);

    let mut boundaries = Vec::with_capacity(word_tokens.len());
    let mut acc = 0usize;
    boundaries.push(0);
    for group in &word_tokens[..word_tokens.len() - 1] {
        acc += group.len();
        boundaries.push(acc);
    }

    // the time of every step where the text index moves, the first step included
    let jump_times: Vec<f64> = (0..text_idx.len())
        .filter(|&k| k == 0 || text_idx[k] != text_idx[k - 1])
        .map(|k| time_idx[k] as f64 / TOKENS_PER_SECOND as f64)
        .collect();
    if boundaries[boundaries.len() - 1] >= jump_times.len() {
        return Vec::new();
    }

    boundaries
        .windows(2)
        .take(n_words)
        .map(|w| WordTime {
            start: jump_times[w[0]],
            end: jump_times[w[1]],
        })
        .collect()
}

/// Masks `neg` to `+inf` outside the audio bands the decoder's own timestamps allow,
/// in place, as `_apply_anchor_bands`
///
/// Word anchors confine their word's rows to the anchor's frames plus `slack_frames`,
/// skipping anchors that don't move forward in time. Tag anchors confine the last row
/// of the word before the tag to end by the tag's end, and the first row of the word
/// after it to start at the tag's start. Errors only when `word_tokens` is shorter
/// than `n_words` needs (where Python raises `IndexError`)
pub fn apply_anchor_bands(
    neg: &mut NegMatrix,
    n_words: usize,
    word_tokens: &[Vec<u32>],
    audio_cap: usize,
    word_anchors: &[WordAnchor],
    tag_anchors: &[TagAnchor],
    slack_frames: i64,
) -> Result<()> {
    let n_rows = neg.rows as i64;
    let audio_cap = audio_cap as i64;
    let last_word = n_words as i64 - 1;

    // first row of each word: row 0 is the no-timestamps token, so the groups start at 1
    let mut anchor_rows = vec![1i64];
    for group in word_tokens.iter().take(word_tokens.len().saturating_sub(1)) {
        anchor_rows.push(anchor_rows[anchor_rows.len() - 1] + group.len() as i64);
    }

    let row = |k: i64| -> Result<i64> {
        usize::try_from(k)
            .ok()
            .and_then(|k| anchor_rows.get(k).copied())
            .ok_or_else(|| eyre!("anchor word {k} outside {} token groups", anchor_rows.len()))
    };
    let frame = |t: f64| (t * TOKENS_PER_SECOND as f64).round_ties_even() as i64;

    // (row range, columns before which to mask, columns from which to mask)
    let mut bands: Vec<(i64, i64, Option<i64>, Option<i64>)> = Vec::new();
    let mut seen_max_audio = -1.0f64;
    for a in word_anchors {
        let wi = a.word as i64;
        if wi >= last_word || a.end <= seen_max_audio {
            continue;
        }

        let a_lo = (frame(a.start) - slack_frames).max(0);
        let a_hi = (frame(a.end) + slack_frames).min(audio_cap);
        if a_lo >= a_hi {
            continue;
        }

        let r_lo = row(wi)?;
        let r_hi = row(wi + 1)?.min(n_rows);
        if r_lo >= r_hi {
            continue;
        }

        bands.push((r_lo, r_hi, Some(a_lo), Some(a_hi)));
        seen_max_audio = a.end;
    }

    for a in tag_anchors {
        let a_lo = (frame(a.start) - slack_frames).max(0);
        let a_hi = (frame(a.end) + slack_frames).min(audio_cap);
        if a_hi <= 0 || a_lo >= audio_cap {
            continue;
        }

        if let Some(p) = a.prev.map(|p| p as i64).filter(|&p| p < last_word) {
            let last_row = (row(p + 1)? - 1).min(n_rows - 1);
            if last_row >= 0 {
                bands.push((last_row, last_row + 1, None, Some(a_hi)));
            }
        }

        if let Some(q) = a.next.map(|q| q as i64).filter(|&q| q < last_word) {
            let first_row = row(q)?.min(n_rows - 1);
            if first_row >= 0 {
                bands.push((first_row, first_row + 1, Some(a_lo), None));
            }
        }
    }

    let (rows, cols) = (neg.rows, neg.cols);
    for (r_lo, r_hi, a_lo, a_hi) in bands {
        let r = py_slice(Some(r_lo), Some(r_hi), rows);
        if let Some(a_lo) = a_lo.filter(|&a| a > 0) {
            neg.fill(r.clone(), py_slice(None, Some(a_lo), cols), f32::INFINITY);
        }

        if let Some(a_hi) = a_hi.filter(|&a| a < cols as i64) {
            neg.fill(r, py_slice(Some(a_hi), None, cols), f32::INFINITY);
        }
    }

    Ok(())
}

/// The index range a Python `seq[start:stop]` selects from a sequence of `len` items
fn py_slice(start: Option<i64>, stop: Option<i64>, len: usize) -> Range<usize> {
    let len = len as i64;
    let clip = |k: i64| if k < 0 { (k + len).max(0) } else { k.min(len) };
    let start = start.map_or(0, clip);
    let stop = stop.map_or(len, clip);
    start as usize..stop.max(start) as usize
}

#[cfg(test)]
mod tests {
    use super::{NegMatrix, dtw_path, dtw_path_with, strip_scalar};

    /// The numba loop as written, column by column
    ///
    /// Returns `(text_idx, time_idx)`, both of the path's length. Indices can be -1 where
    /// the path runs along the matrix's top or left edge
    fn dtw_path_reference(x: &NegMatrix) -> (Vec<i64>, Vec<i64>) {
        let (n, m) = (x.rows, x.cols);
        let h = n + 1;

        // column-major (j outer, i inner, as the numba loop runs) so the inner loop is
        // contiguous; the arithmetic and the comparison order are the kernel's
        let mut cost = vec![f32::INFINITY; (m + 1) * h];
        let mut trace = vec![0i8; (m + 1) * h];
        // row 0 moves left, column 0 moves up
        trace.iter_mut().step_by(h).for_each(|t| *t = 2);
        trace[..h].fill(1);

        trace[0] = -1;
        cost[0] = 0.0;

        let mut xt = vec![0f32; n * m];
        for i in 0..n {
            for j in 0..m {
                xt[j * n + i] = x.data[i * m + j];
            }
        }

        for j in 1..=m {
            for i in 1..=n {
                let c0 = cost[(j - 1) * h + i - 1];
                let c1 = cost[j * h + i - 1];
                let c2 = cost[(j - 1) * h + i];
                // numba compiles without fastmath, so these are IEEE comparisons (false on NaN)
                let (c, t) = if c0 <= c1 && c0 <= c2 {
                    (c0, 0)
                } else if c1 <= c2 {
                    (c1, 1)
                } else {
                    (c2, 2)
                };
                cost[j * h + i] = xt[(j - 1) * n + i - 1] + c;
                trace[j * h + i] = t;
            }
        }

        let mut text_idx = Vec::with_capacity(n + m);
        let mut time_idx = Vec::with_capacity(n + m);
        let (mut i, mut j) = (n, m);
        while i > 0 || j > 0 {
            text_idx.push(i as i64 - 1);
            time_idx.push(j as i64 - 1);
            match trace[j * h + i] {
                0 => {
                    i -= 1;
                    j -= 1;
                }
                1 => i -= 1,
                _ => j -= 1,
            }
        }

        text_idx.reverse();
        time_idx.reverse();
        (text_idx, time_idx)
    }

    #[test]
    fn wavefront_matches_numba_loop() {
        // xorshift, so the cases are fixed
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for case in 0..3000 {
            let n = (next() % 40) as usize;
            let m = (next() % 70) as usize;
            // few distinct values, so ties decide most moves; some NaN and infinities
            let data: Vec<f32> = (0..n * m)
                .map(|_| match next() % 16 {
                    0 => f32::NAN,
                    1 => f32::INFINITY,
                    2 => f32::NEG_INFINITY,
                    r if case % 2 == 0 => -((r % 3) as f32),
                    _ => -((next() >> 40) as f32 / 16_777_216.0),
                })
                .collect();
            let x = NegMatrix::new(n, m, data).unwrap();
            let want = dtw_path_reference(&x);
            assert_eq!(dtw_path(&x), want, "case {case}: {n}x{m}");
            assert_eq!(
                dtw_path_with(&x, strip_scalar),
                want,
                "case {case}: {n}x{m} scalar"
            );
        }
    }
}
