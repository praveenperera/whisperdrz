//! The CPU side of the ECAPA2 speaker remap in `transcriber.py`: which audio each
//! line is embedded from (`extract_line_audio_segments`), clustering the line
//! embeddings, and rewriting the `[N]` tags (`remap_speakers_per_line`)

use std::collections::HashMap;

use eyre::Result;

use super::cluster::{Clustering, cluster_speakers};
use super::pyrandom::CropRng;
use crate::post::py;
use crate::post::segments::{Speaker, parse_to_segments};
use crate::post::speakers::SPEAKER_LINE_START;
use crate::post::timings::TIMINGS_FORMAT;

/// Sample rate of the audio the crops index
const SAMPLE_RATE: f64 = 16000.0;
/// ECAPA2 embedding width
pub const EMBED_DIM: usize = 192;

/// One line's audio for the speaker embedder, as a range of the file's samples
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineCrop {
    /// Index into `parse_to_segments(text)`
    pub line: usize,
    /// The model's speaker id for the line (never `c`)
    pub speaker: i64,
    /// First sample
    pub start: usize,
    /// Number of samples
    pub len: usize,
}

/// Lines shorter than this are not embedded (`min_duration` of the remap)
pub const MIN_CROP_SECONDS: f64 = 1.0;
/// Lines longer than this are embedded from a random window (`max_duration`)
pub const MAX_CROP_SECONDS: f64 = 10.0;

/// The samples a segment's crop comes from, before any random window
enum LineSpan {
    /// The whole span is the crop: no random draw
    Fixed { start: i64, end: i64 },
    /// Longer than `max_duration`: the crop is a random window of the span
    Random { start: i64, duration: i64 },
}

/// The span of one segment in a file of `audio_len` samples, `None` for crowd lines
/// and lines under `min_duration`
fn line_span(
    audio_len: usize,
    seg: &crate::post::Segment,
    min_duration: f64,
    max_duration: f64,
) -> Option<(i64, LineSpan)> {
    let Speaker::Id(speaker) = seg.speaker else {
        return None;
    };
    // `int(max(0, start) * 16000)`: `int()` truncates toward zero
    let s = (py::max(0.0, seg.start) * SAMPLE_RATE) as i64;
    let mut e = (seg.end * SAMPLE_RATE) as i64;
    if e > audio_len as i64 {
        e = audio_len as i64;
    }

    let duration = e - s;
    if duration < (min_duration * SAMPLE_RATE) as i64 {
        return None;
    }

    let span = if duration > (max_duration * SAMPLE_RATE) as i64 {
        LineSpan::Random { start: s, duration }
    } else {
        LineSpan::Fixed { start: s, end: e }
    };
    Some((speaker, span))
}

/// `extract_line_audio_segments(audio, text, min_duration, max_duration)` over a
/// file of `audio_len` samples: one crop per non-crowd segment lasting at least
/// `min_duration`; segments longer than `max_duration` get a random window of 3 s to
/// `max_duration`, drawn from `rng` (the global `random` in Python, seeded per file)
pub fn extract_line_audio_segments(
    audio_len: usize,
    text: &str,
    min_duration: f64,
    max_duration: f64,
    rng: &mut (impl CropRng + ?Sized),
) -> Result<Vec<LineCrop>> {
    const MIN_WINDOW: f64 = 3.0;

    let mut out = Vec::new();
    for (line, seg) in parse_to_segments(text)?.iter().enumerate() {
        let Some((speaker, span)) = line_span(audio_len, seg, min_duration, max_duration) else {
            continue;
        };

        let (s, e) = match span {
            LineSpan::Fixed { start, end } => (start, end),
            LineSpan::Random { start, duration } => {
                let window =
                    ((rng.uniform(MIN_WINDOW, max_duration) * SAMPLE_RATE) as i64).min(duration);
                let s = start + rng.randint(0, duration - window)?;
                (s, s + window)
            }
        };

        out.push(LineCrop {
            line,
            speaker,
            start: s as usize,
            len: (e - s) as usize,
        });
    }

    Ok(out)
}

/// The crops of [`extract_line_audio_segments`] that draw nothing from the rng, as
/// `(start, len)`: a line's crop depends on the rng only when the line is longer than
/// `max_duration`, so these are exact from the line's own text
pub fn fixed_line_crops(
    audio_len: usize,
    text: &str,
    min_duration: f64,
    max_duration: f64,
) -> Result<Vec<(usize, usize)>> {
    Ok(parse_to_segments(text)?
        .iter()
        .filter_map(
            |seg| match line_span(audio_len, seg, min_duration, max_duration)? {
                (_, LineSpan::Fixed { start, end }) => {
                    Some((start as usize, (end - start) as usize))
                }
                (_, LineSpan::Random { .. }) => None,
            },
        )
        .collect())
}

/// What a remap did, for logging and parity checks
#[derive(Debug, Clone, PartialEq)]
pub struct Remap {
    pub text: String,
    pub crops: Vec<LineCrop>,
    /// `None` when the remap returned the text unchanged before clustering
    pub clustering: Option<Clustering>,
}

/// `remap_speakers_per_line(text, audio, model, threshold, min_duration)`.
///
/// `embed(start, len)` embeds `len` samples from `start` with ECAPA2 and returns
/// the L2-normalized f32 vector `_embed_audio_segment` returns; it runs once per crop,
/// in line order. Lines are clustered from scratch; lines too short to embed take
/// the cluster of the nearest embedded line with the same model speaker, or a new
/// cluster per model speaker when none exists
pub fn remap_speakers_per_line(
    text: &str,
    audio_len: usize,
    rng: &mut impl CropRng,
    mut embed: impl FnMut(usize, usize) -> Result<[f32; EMBED_DIM]>,
    threshold: f64,
    min_duration: f64,
) -> Result<Remap> {
    let unchanged = |crops| Remap {
        text: text.to_owned(),
        crops,
        clustering: None,
    };

    let segments = parse_to_segments(text)?;
    if segments.is_empty() {
        return Ok(unchanged(Vec::new()));
    }

    let crops = extract_line_audio_segments(audio_len, text, min_duration, MAX_CROP_SECONDS, rng)?;
    if crops.is_empty() {
        return Ok(unchanged(crops));
    }

    let mut embeddings = Vec::with_capacity(crops.len());
    for c in &crops {
        embeddings.push(embed(c.start, c.len)?.to_vec());
    }

    if embeddings.len() < 2 {
        return Ok(unchanged(crops));
    }

    let clustering = cluster_speakers(&embeddings, threshold, 1)?;
    let embedded: HashMap<usize, i64> = crops
        .iter()
        .zip(&clustering.labels)
        .map(|(c, &l)| (c.line, l))
        .collect();
    // crops come out in line order, so this is `sorted(embedded_set)` with starts
    let embedded_info: Vec<(usize, f64)> = crops
        .iter()
        .map(|c| (c.line, segments[c.line].start))
        .collect();

    let mut model_speaker: HashMap<usize, Speaker> = crops
        .iter()
        .map(|c| (c.line, Speaker::Id(c.speaker)))
        .collect();
    for (i, seg) in segments.iter().enumerate() {
        if seg.speaker != Speaker::Crowd {
            model_speaker.entry(i).or_insert(seg.speaker);
        }
    }

    let mut max_cluster = clustering.labels.iter().copied().max().unwrap_or(-1);
    let mut unembedded: Vec<(Speaker, i64)> = Vec::new();
    let mut line_speaker: HashMap<usize, i64> = HashMap::new();
    for (i, seg) in segments.iter().enumerate() {
        if seg.speaker == Speaker::Crowd {
            continue;
        }

        if let Some(&c) = embedded.get(&i) {
            line_speaker.insert(i, c);
            continue;
        }

        let mut best: Option<(f64, usize)> = None;
        for &(ei, ei_start) in &embedded_info {
            if model_speaker.get(&ei) == Some(&seg.speaker) {
                let dist = (seg.start - ei_start).abs();
                if best.is_none_or(|(d, _)| dist < d) {
                    best = Some((dist, ei));
                }
            }
        }

        let cluster = match best {
            Some((_, ei)) => embedded[&ei],
            None => match unembedded.iter().find(|(s, _)| *s == seg.speaker) {
                Some(&(_, c)) => c,
                None => {
                    max_cluster += 1;
                    unembedded.push((seg.speaker, max_cluster));
                    max_cluster
                }
            },
        };
        line_speaker.insert(i, cluster);
    }

    let mut out = Vec::new();
    let mut seg_idx = 0;
    for line in text.split('\n') {
        let stripped = py::strip(line);
        if stripped.is_empty() || stripped.starts_with("[CHOP]") {
            out.push(line.to_owned());
            continue;
        }

        let Some(m) = SPEAKER_LINE_START.search(stripped)? else {
            out.push(line.to_owned());
            continue;
        };

        // the counter follows lines with two or more markers, which can drift from
        // `segments` when a line ends no later than it starts; Python has the same drift
        let timed = TIMINGS_FORMAT.captures_all(stripped)?.len() >= 2;
        let spk = m.group(1);
        if spk == "c" {
            out.push(line.to_owned());
            if timed && seg_idx < segments.len() && segments[seg_idx].speaker == Speaker::Crowd {
                seg_idx += 1;
            }

            continue;
        }

        if timed && seg_idx < segments.len() {
            let new_spk = match line_speaker.get(&seg_idx) {
                Some(&c) => c,
                None => py::int(spk)?,
            };
            let rest = &stripped[m.end()..];
            out.push(if rest.is_empty() {
                format!("[{new_spk}]")
            } else {
                format!("[{new_spk}] {rest}")
            });
            seg_idx += 1;
        } else {
            out.push(line.to_owned());
        }
    }

    Ok(Remap {
        text: out.join("\n"),
        crops,
        clustering: Some(clustering),
    })
}
