//! The batched entry point: per-slot prep, the padded token batch, per-chunk DTW and
//! anchor stitching (`fill_word_timings_batch`, `_build_full_tokens_batch`)

use eyre::{Result, ensure, eyre};

use super::chunk::{CharTokens, ChunkPrep, chunk_speaker_lines};
use super::kernel::{NegMatrix, WordTime, apply_anchor_bands, dtw_from_neg_matrix};
use super::parse::{ParsedText, parse_predicted_lines};
use super::rebuild::rebuild_with_timings;
use super::{DtwTokenizer, SLACK_FRAMES};

/// Knobs `fill_word_timings_batch` takes from its caller
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DtwOptions {
    /// Char-level token budget per chunk, below the decoder's 448 positions
    pub max_tokens: usize,
    /// Encoder positions per window (`xa.shape[0]`), which caps each chunk's audio
    pub n_audio_ctx: usize,
}

impl Default for DtwOptions {
    fn default() -> Self {
        Self {
            max_tokens: 440,
            n_audio_ctx: 1500,
        }
    }
}

/// The GPU forward's input: one row per pooled chunk
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DtwBatch {
    /// `[rows, t_pad]` row-major: sot sequence, no-timestamps, the chunk's char tokens,
    /// eot, then eot padding
    pub full_tokens: Vec<u32>,
    pub t_pad: usize,
    /// Unpadded length of each row
    pub actual_lens: Vec<usize>,
    /// Audio positions each row's softmax may attend to (`num_frames // 2`, capped)
    pub audio_caps: Vec<usize>,
    /// The slot whose encoder output (`xa`) each row attends to
    pub slots: Vec<usize>,
}

impl DtwBatch {
    pub fn rows(&self) -> usize {
        self.actual_lens.len()
    }

    /// Width of the prepped matrices
    pub fn max_audio_cap(&self) -> usize {
        self.audio_caps.iter().copied().max().unwrap_or(0)
    }

    /// One row's tokens, padding included
    pub fn row(&self, bi: usize) -> &[u32] {
        &self.full_tokens[bi * self.t_pad..(bi + 1) * self.t_pad]
    }
}

/// The GPU prep's output (`_dtw_prep_batch`): `[rows, t_pad, max_audio_cap]` row-major
/// normalised, median-filtered, head-averaged attention
#[derive(Debug, Clone, PartialEq)]
pub struct Matrices {
    pub shape: [usize; 3],
    pub data: Vec<f32>,
}

/// Per flat word of a slot, its stitched `(start, end)`
type SlotTimings = Vec<Option<(f64, f64)>>;

/// Per word of a chunk, its DTW times
type ChunkTimes = Vec<WordTime>;

/// One pooled chunk: the slot it came from and its prep
#[derive(Debug, Clone)]
struct Pooled {
    slot: usize,
    rec: ChunkPrep,
}

/// A `fill_word_timings_batch` call split into its stages, so tests and the GPU
/// runtime can step through it
#[derive(Debug, Clone)]
pub struct PreparedBatch<'a> {
    texts: Vec<&'a str>,
    parsed: Vec<ParsedText>,
    pooled: Vec<Pooled>,
    batch: Option<DtwBatch>,
    sot_len: usize,
}

impl<'a> PreparedBatch<'a> {
    /// Parses and chunks every slot and builds the token batch; `slots` are
    /// `(window_text, num_frames)`
    pub fn new<S, T>(slots: &'a [(S, i64)], tokenizer: &T, opts: &DtwOptions) -> Result<Self>
    where
        S: AsRef<str>,
        T: DtwTokenizer + ?Sized,
    {
        let mut tokens = CharTokens::new(tokenizer);
        let mut parsed = Vec::with_capacity(slots.len());
        let mut pooled = Vec::new();
        let mut audio_caps = Vec::new();
        for (si, (text, num_frames)) in slots.iter().enumerate() {
            let p = parse_predicted_lines(text.as_ref())?;
            if !p.words.is_empty() {
                let chunks = chunk_speaker_lines(&p.lines, &p.words, &mut tokens, opts.max_tokens);
                for chunk in &chunks {
                    let Some(rec) = ChunkPrep::new(
                        chunk,
                        &mut tokens,
                        &p.words,
                        &p.tag_anchors,
                        opts.max_tokens,
                    ) else {
                        continue;
                    };
                    pooled.push(Pooled { slot: si, rec });
                    // `max(0, min(num_frames // 2, T_audio))` with floor division
                    audio_caps.push(num_frames.div_euclid(2).min(opts.n_audio_ctx as i64).max(0) as usize);
                }
            }

            parsed.push(p);
        }

        let sot = tokenizer.sot_sequence();
        let batch = (!pooled.is_empty()).then(|| {
            let (eot, no_ts) = (tokenizer.eot(), tokenizer.no_timestamps());
            let actual_lens: Vec<usize> = pooled
                .iter()
                .map(|p| sot.len() + p.rec.ids.len() + 2)
                .collect();
            let t_pad = actual_lens.iter().copied().max().unwrap_or(0);
            let mut full_tokens = vec![eot; pooled.len() * t_pad];
            for (bi, p) in pooled.iter().enumerate() {
                let row = &mut full_tokens[bi * t_pad..];
                let seq = sot
                    .iter()
                    .copied()
                    .chain([no_ts])
                    .chain(p.rec.ids.iter().copied())
                    .chain([eot]);
                for (dst, t) in row.iter_mut().zip(seq) {
                    *dst = t;
                }
            }

            DtwBatch {
                full_tokens,
                t_pad,
                actual_lens,
                audio_caps,
                slots: pooled.iter().map(|p| p.slot).collect(),
            }
        });

        let texts = slots.iter().map(|(t, _)| t.as_ref()).collect();
        Ok(Self {
            texts,
            parsed,
            pooled,
            batch,
            sot_len: sot.len(),
        })
    }

    /// The GPU forward's input, or `None` when no slot has a chunk to align
    pub fn batch(&self) -> Option<&DtwBatch> {
        self.batch.as_ref()
    }

    /// Pooled chunk `bi`'s prep
    pub fn chunk(&self, bi: usize) -> &ChunkPrep {
        &self.pooled[bi].rec
    }

    /// Checks the prep output's shape against the batch
    pub fn check_matrices(&self, m: &Matrices) -> Result<()> {
        let batch = self
            .batch
            .as_ref()
            .ok_or_else(|| eyre!("matrices for an empty batch"))?;
        let want = [batch.rows(), batch.t_pad, batch.max_audio_cap()];
        ensure!(
            m.shape == want,
            "prep returned shape {:?}, want {want:?}",
            m.shape
        );
        ensure!(
            m.data.len() == want.iter().product::<usize>(),
            "prep data length {} != {want:?}",
            m.data.len()
        );
        Ok(())
    }

    /// Chunk `bi`'s cost matrix: its text rows (no-timestamps row first, eot row
    /// dropped) and real audio columns, negated and anchor-banded
    pub fn chunk_neg_matrix(&self, bi: usize, m: &Matrices) -> Result<NegMatrix> {
        let batch = self.batch.as_ref().ok_or_else(|| eyre!("no batch"))?;
        let rec = &self.pooled[bi].rec;
        let [_, t_pad, width] = m.shape;
        let audio_cap = batch.audio_caps[bi];
        let rows = self.sot_len.min(t_pad)..(batch.actual_lens[bi] - 1).min(t_pad);
        let cols = audio_cap.min(width);

        let mut data = Vec::with_capacity(rows.len() * cols);
        for r in rows.clone() {
            let base = (bi * t_pad + r) * width;
            data.extend(m.data[base..base + cols].iter().map(|&v| -v));
        }

        let mut neg = NegMatrix::new(rows.len(), cols, data)?;
        if !neg.is_empty() && (!rec.word_anchors.is_empty() || !rec.tag_anchors.is_empty()) {
            apply_anchor_bands(
                &mut neg,
                rec.ext_words.len(),
                &rec.ext_word_tokens,
                audio_cap,
                &rec.word_anchors,
                &rec.tag_anchors,
                SLACK_FRAMES,
            )?;
        }

        Ok(neg)
    }

    /// DTW over chunk `bi`'s cost matrix
    pub fn chunk_dtw(&self, bi: usize, neg: &NegMatrix) -> Vec<WordTime> {
        let rec = &self.pooled[bi].rec;
        dtw_from_neg_matrix(neg, rec.ext_words.len(), &rec.ext_word_tokens)
    }

    /// The input texts, for a call with nothing to align
    pub fn unchanged(&self) -> Vec<String> {
        self.texts.iter().map(|t| (*t).to_owned()).collect()
    }

    /// Stitches each slot's chunks onto its window timeline and rewrites the texts;
    /// `results[bi]` is chunk `bi`'s DTW
    ///
    /// A slot's first chunk keeps DTW's own times. Later chunks shift so their
    /// earliest decoder-timed word lands on the decoder's start, or, with no such
    /// word, so their first word starts where the previous chunk ended. Slots where
    /// no chunk produced timings come back unchanged
    pub fn finish(&self, results: &[Vec<WordTime>]) -> Result<Vec<String>> {
        ensure!(
            results.len() == self.pooled.len(),
            "{} DTW results for {} chunks",
            results.len(),
            self.pooled.len()
        );

        let n_slots = self.texts.len();
        let mut out_timings: Vec<Option<SlotTimings>> = vec![None; n_slots];
        let mut prev_end: Vec<Option<f64>> = vec![None; n_slots];
        let mut chunk_count = vec![0usize; n_slots];

        for (Pooled { slot: si, rec }, dtw) in self.pooled.iter().zip(results) {
            let si = *si;
            if dtw.is_empty() {
                chunk_count[si] += 1;
                continue;
            }

            let flat = &self.parsed[si].words;
            let tagged = &rec.tagged;
            let mut word_t: Vec<Option<(f64, f64)>> = vec![None; tagged.len()];
            for (slot, wt) in word_t.iter_mut().zip(dtw) {
                *slot = Some((wt.start, wt.end));
            }

            // real words with a DTW time, in chunk order
            let real = || {
                tagged
                    .iter()
                    .zip(&word_t)
                    .filter_map(|(oi, t)| Some(((*oi)?, (*t)?)))
            };
            let anchor = real().find_map(|(oi, t)| flat[oi].start.map(|dec_s| (dec_s, t)));

            let shift = if chunk_count[si] == 0 {
                0.0
            } else if let Some((dec_s, t)) = anchor {
                dec_s - t.0
            } else if let Some(prev) = prev_end[si] {
                real().next().map_or(0.0, |(_, t)| prev - t.0)
            } else {
                0.0
            };

            let slot_out = out_timings[si].get_or_insert_with(|| vec![None; flat.len()]);
            let mut last_end = None;
            for (oi, (s, e)) in real() {
                let e = e + shift;
                slot_out[oi] = Some((s + shift, e));
                last_end = Some(e);
            }

            if last_end.is_some() {
                prev_end[si] = last_end;
            }

            chunk_count[si] += 1;
        }

        self.texts
            .iter()
            .zip(&self.parsed)
            .zip(out_timings)
            .map(|((text, parsed), timings)| match timings {
                Some(t) => rebuild_with_timings(text, &parsed.words, &t),
                None => Ok((*text).to_owned()),
            })
            .collect()
    }
}

/// Fills in word timings for every finalising window, as `dtw_align.fill_word_timings_batch`
///
/// `slots` are `(window_text, num_frames)`. `prep` runs the GPU part on the batch (the
/// decoder forward over `full_tokens` with each row attending to its slot's encoder
/// output, cross-attention capture over the alignment heads, then `_dtw_prep_batch`)
/// and is called at most once, not at all when nothing needs aligning. Returns one
/// text per slot, unchanged where there was nothing to fill
pub fn fill_word_timings_batch<S, T>(
    slots: &[(S, i64)],
    tokenizer: &T,
    opts: &DtwOptions,
    mut prep: impl FnMut(&DtwBatch) -> Result<Matrices>,
) -> Result<Vec<String>>
where
    S: AsRef<str>,
    T: DtwTokenizer + ?Sized,
{
    let prepared = PreparedBatch::new(slots, tokenizer, opts)?;
    let Some(batch) = prepared.batch() else {
        return Ok(prepared.unchanged());
    };

    let matrices = prep(batch)?;
    prepared.check_matrices(&matrices)?;

    // chunks are independent, so they spread over threads like Python's pool of 8;
    // stitching below runs in chunk order
    let n = batch.rows();
    let workers = std::thread::available_parallelism()
        .map_or(1, |p| p.get())
        .clamp(1, 8)
        .min(n);
    let per_worker: Vec<Result<Vec<(usize, ChunkTimes)>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|w| {
                let (prepared, matrices) = (&prepared, &matrices);
                scope.spawn(move || {
                    (w..n)
                        .step_by(workers)
                        .map(|bi| {
                            Ok((
                                bi,
                                prepared.chunk_dtw(bi, &prepared.chunk_neg_matrix(bi, matrices)?),
                            ))
                        })
                        .collect()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(eyre!("DTW worker panicked")))
            })
            .collect()
    });

    let mut results = vec![Vec::new(); n];
    for done in per_worker {
        for (bi, r) in done? {
            results[bi] = r;
        }
    }

    prepared.finish(&results)
}
