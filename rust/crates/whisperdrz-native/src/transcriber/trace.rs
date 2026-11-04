//! NVTX ranges over the scheduler's host phases, so nsys charges GPU idle time to the
//! host work that caused it. Without the `cuda` feature every range is a no-op

#[cfg(feature = "cuda")]
pub(crate) use crate::cuda::nvtx::Range;

/// The no-op range of a build without the GPU runtime
#[cfg(not(feature = "cuda"))]
#[must_use = "the range closes when this is dropped"]
pub(crate) struct Range;

#[cfg(not(feature = "cuda"))]
impl Range {
    pub(crate) fn new(_name: &str) -> Self {
        Self
    }
}
