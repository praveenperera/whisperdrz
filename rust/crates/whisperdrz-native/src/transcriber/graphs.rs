//! Bucket choice and padding for the decode step (`BucketedCUDAGraphs.forward` with
//! the `--pad-fix` trash slot)

/// `CUDA_GRAPH_BUCKETS`: batch sizes a decode graph is captured for
pub const CUDA_GRAPH_BUCKETS: [usize; 9] = [1, 2, 4, 8, 16, 32, 42, 52, 54];

/// `BucketedCUDAGraphs.capture_all(max_bucket)`: the buckets at or under
/// `max_bucket`, plus `max_bucket` itself when it isn't one
pub fn captured_buckets(max_bucket: usize) -> Vec<usize> {
    let mut buckets: Vec<usize> = CUDA_GRAPH_BUCKETS
        .iter()
        .copied()
        .filter(|&b| b <= max_bucket)
        .collect();
    if !buckets.contains(&max_bucket) {
        buckets.push(max_bucket);
        buckets.sort_unstable();
    }

    buckets
}

/// `_get_bucket`: the smallest bucket that fits `batch`, or `batch` itself when
/// none does (which runs the decoder directly, unpadded)
pub fn get_bucket(batch: usize, buckets: &[usize]) -> usize {
    buckets
        .iter()
        .copied()
        .find(|&b| batch <= b)
        .unwrap_or(batch)
}

/// One decode step's inputs, padded to the bucket
///
/// The first `rows` entries are the real rows: each slot's last written token, its
/// KV length as the position, and the slot. The rest (up to `bucket`) are padding:
/// token 0, position 0, the trash slot
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeStep {
    pub rows: usize,
    pub bucket: usize,
    pub tokens: Vec<u32>,
    pub positions: Vec<i32>,
    pub slots: Vec<usize>,
    /// The KV slot padding rows write to (`self_kv[0].batch_size - 1`)
    pub trash_slot: usize,
}

impl DecodeStep {
    /// Pads `tokens`, `positions` and `slots` (one entry per decoding slot) to the
    /// bucket for their length
    pub fn new(
        tokens: Vec<u32>,
        positions: Vec<i32>,
        slots: Vec<usize>,
        buckets: &[usize],
        trash_slot: usize,
    ) -> Self {
        let rows = slots.len();
        debug_assert!(tokens.len() == rows && positions.len() == rows);
        let bucket = get_bucket(rows, buckets);
        let mut step = Self {
            rows,
            bucket,
            tokens,
            positions,
            slots,
            trash_slot,
        };
        step.tokens.resize(bucket, 0);
        step.positions.resize(bucket, 0);
        step.slots.resize(bucket, trash_slot);
        step
    }

    /// Whether the step runs a captured graph with padding rows, which is when the
    /// trash slot's KV length has to be zeroed first
    pub fn is_padded(&self) -> bool {
        self.bucket > self.rows
    }
}
