//! The CPU side of the ECAPA2 speaker remap: CPython's `random` for the line crops,
//! clustering in torch's f32 CPU arithmetic, and the `[N]` tag rewrite. The ECAPA2
//! forward pass itself comes in as a closure

pub mod cluster;
pub mod pyrandom;
pub mod remap;
pub mod torch_cpu;

pub use cluster::{Clustering, MergeStep, cluster_speakers};
pub use pyrandom::{CropRng, PyRandom};
pub use remap::{
    EMBED_DIM, LineCrop, MAX_CROP_SECONDS, MIN_CROP_SECONDS, Remap, extract_line_audio_segments,
    fixed_line_crops, remap_speakers_per_line,
};
