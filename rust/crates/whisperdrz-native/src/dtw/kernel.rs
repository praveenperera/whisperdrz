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

/// Minimum-cost monotone path through `x`, as `_dtw_numba`
///
/// Returns `(text_idx, time_idx)`, both of the path's length. Indices can be -1 where
/// the path runs along the matrix's top or left edge
pub fn dtw_path(x: &NegMatrix) -> (Vec<i64>, Vec<i64>) {
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
