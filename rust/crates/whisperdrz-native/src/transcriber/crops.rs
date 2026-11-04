//! Early crop planning for the speaker remap: as chunks finish, works out which
//! crops `remap_speakers_per_line` will embed and hands them to the model to embed
//! alongside decoding
//!
//! The remap runs on `join_predictions` of every chunk before any merge or repair,
//! and `parse_to_segments` reads each line on its own, so a chunk's lines and their
//! crops never depend on later chunks. A line of at most `MAX_CROP_SECONDS` is
//! cropped from its own times, so its crop is exact once its chunk finishes. A
//! longer line takes a random window, and each draw advances the rng for every later
//! long line, so those crops are extracted chunk by chunk in order with a clone of
//! the file's rng, as soon as every earlier chunk is in. The prefetched crops are
//! therefore exactly the remap's crops, and none is wasted

use std::sync::Arc;

use eyre::{Result, ensure};

use super::Model;
use crate::post::join_predictions;
use crate::speaker::{
    CropRng, MAX_CROP_SECONDS, MIN_CROP_SECONDS, extract_line_audio_segments, fixed_line_crops,
};

/// One `transcribe` call's crop planning
pub(crate) struct CropPlanner {
    audio: Arc<[f32]>,
    /// The file's rng as `transcribe` received it, advanced past the long lines of
    /// chunks `0..next`
    rng: Box<dyn CropRng>,
    /// Each finished chunk's joined text, by chunk number
    texts: Vec<Option<String>>,
    /// Chunks before this have had every crop extracted
    next: usize,
}

impl CropPlanner {
    pub(crate) fn new(audio: Arc<[f32]>, rng: Box<dyn CropRng>, chunks: usize) -> Self {
        Self {
            audio,
            rng,
            texts: vec![None; chunks],
            next: 0,
        }
    }

    /// Chunk `index` finished with `prediction`: prefetches its rng-free crops, then
    /// every crop of the chunks the in-order prefix now covers
    pub(crate) fn finished(
        &mut self,
        index: usize,
        prediction: &str,
        model: &mut impl Model,
    ) -> Result<()> {
        ensure!(
            self.texts.get(index).is_some_and(Option::is_none),
            "chunk {index} of {} finished twice or out of range",
            self.texts.len()
        );
        // `join_predictions` of one chunk is that chunk's part of the joined text
        let text = join_predictions([prediction]);
        let audio_len = self.audio.len();
        for (start, len) in fixed_line_crops(audio_len, &text, MIN_CROP_SECONDS, MAX_CROP_SECONDS)?
        {
            model.prefetch_embed(&self.audio, start, len)?;
        }

        self.texts[index] = Some(text);
        while let Some(Some(text)) = self.texts.get(self.next) {
            let crops = extract_line_audio_segments(
                audio_len,
                text,
                MIN_CROP_SECONDS,
                MAX_CROP_SECONDS,
                &mut *self.rng,
            )?;
            // the rng-free crops were prefetched above; the model drops repeats
            for c in crops {
                model.prefetch_embed(&self.audio, c.start, c.len)?;
            }

            self.next += 1;
        }

        Ok(())
    }
}
