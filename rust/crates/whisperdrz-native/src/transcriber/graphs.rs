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

/// One speculative verify step's inputs: rows of different lengths, packed
///
/// Row `r` feeds `tokens[starts[r]..starts[r + 1]]`, its slot's last written token then
/// its drafts, at consecutive positions from the slot's KV length. Logits come back only
/// for the tokens in `need`: a draft the FSM forces is decided without logits, so the
/// position before it needs none, as long as every draft before it was forced too. The
/// real tokens are followed by padding up to the bucket for their count: token 0 at
/// position 0 on the trash slot, one padding row per token
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyStep {
    /// Real rows
    pub rows: usize,
    /// KV slot of each real row
    pub slots: Vec<usize>,
    /// `rows + 1` offsets into `tokens`; `starts[rows]` is the real token count
    pub starts: Vec<usize>,
    /// `[bucket]` token ids, real then padding
    pub tokens: Vec<u32>,
    /// `[bucket]` positions
    pub positions: Vec<i32>,
    /// The KV slot padding rows write to
    pub trash_slot: usize,
    /// Real tokens whose logits the step returns, ascending; logits row `i` is token
    /// `need[i]`'s
    pub need: Vec<usize>,
    /// Rows the LM head runs over: `need` padded to the bucket for its count, as a
    /// one-token step over that many rows pads
    pub head_rows: usize,
}

/// One row of a [`VerifyStep`]
#[derive(Debug, Clone, Copy)]
pub struct VerifyRow<'a> {
    pub slot: usize,
    /// The slot's KV length, the first fed token's position
    pub kv_len: usize,
    /// The last written token, then the drafts
    pub fed: &'a [u32],
    /// Leading fed tokens after which the next token is forced for certain: their
    /// logits are never read
    pub forced: usize,
}

impl VerifyStep {
    /// Packs `rows` and pads them to the bucket for their token count
    pub fn new(rows: &[VerifyRow<'_>], buckets: &[usize], trash_slot: usize) -> Self {
        let mut slots = Vec::with_capacity(rows.len());
        let mut starts = Vec::with_capacity(rows.len() + 1);
        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        let mut need = Vec::new();
        starts.push(0);
        for row in rows {
            debug_assert!(
                !row.fed.is_empty() && row.forced < row.fed.len(),
                "a verify row feeds at least its last token, and needs the logits of its \
                 last fed token"
            );
            slots.push(row.slot);
            let at = tokens.len();
            need.extend(at + row.forced..at + row.fed.len());
            tokens.extend_from_slice(row.fed);
            positions.extend((0..row.fed.len()).map(|j| (row.kv_len + j) as i32));
            starts.push(tokens.len());
        }

        let bucket = get_bucket(tokens.len(), buckets);
        tokens.resize(bucket, 0);
        positions.resize(bucket, 0);
        let head_rows = get_bucket(need.len(), buckets);
        Self {
            rows: rows.len(),
            slots,
            starts,
            tokens,
            positions,
            trash_slot,
            need,
            head_rows,
        }
    }

    /// The logits row of real token `j`, if the step returns one
    pub fn logit_row(&self, j: usize) -> Option<usize> {
        self.need.binary_search(&j).ok()
    }

    /// Real tokens, padding excluded
    pub fn real_tokens(&self) -> usize {
        self.starts[self.rows]
    }

    /// Tokens including padding
    pub fn bucket(&self) -> usize {
        self.tokens.len()
    }

    /// Tokens row `r` feeds
    pub fn row_len(&self, r: usize) -> usize {
        self.starts[r + 1] - self.starts[r]
    }

    /// The most tokens any row feeds
    pub fn max_row_len(&self) -> usize {
        (0..self.rows).map(|r| self.row_len(r)).max().unwrap_or(0)
    }
}
