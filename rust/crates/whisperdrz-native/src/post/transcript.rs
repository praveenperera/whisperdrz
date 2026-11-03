//! The tail of `Transcriber.transcribe`: join the chunk predictions, remap speakers,
//! merge, repair and renumber, then parse the segments

use eyre::Result;
use serde::Serialize;

use super::segments::{Segment, parse_to_segments};
use super::speakers::{merge_consecutive_speakers, normalize_speaker_ids};
use super::timings::fix_nonmonotonic_timestamps;

/// `TranscribeResults` without `raw_chunks`
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Transcript {
    pub text: String,
    pub segments: Vec<Segment>,
}

/// Joins the chunk predictions in chunk order: each one with `[CHOP]` turned into a
/// newline and trailing whitespace removed, one per line
pub fn join_predictions<'a>(predictions: impl IntoIterator<Item = &'a str>) -> String {
    predictions
        .into_iter()
        .map(|p| {
            p.replace("[CHOP]", "\n")
                .trim_end_matches(super::py::is_space)
                .to_owned()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Merges same-speaker lines, repairs backwards markers and renumbers speakers by
/// first appearance
pub fn finish_text(combined: &str) -> Result<String> {
    let merged = merge_consecutive_speakers(combined)?;
    let merged = fix_nonmonotonic_timestamps(&merged)?;
    normalize_speaker_ids(&merged)
}

/// The whole tail: `remap` is the per-line speaker remap
/// ([`crate::speaker::remap_speakers_per_line`] bound to the file's audio, embedder
/// and random state), or the identity when the remap model is off
pub fn finish(
    predictions: &[&str],
    remap: impl FnOnce(String) -> Result<String>,
) -> Result<Transcript> {
    let combined = remap(join_predictions(predictions.iter().copied()))?;
    let text = finish_text(&combined)?;
    let segments = parse_to_segments(&text)?;
    Ok(Transcript { text, segments })
}
