//! Decoder attention on the GPU: the per-layer KV caches, the per-slot encoder outputs,
//! and the cuda-oxide kernels that attend over them
//!
//! The reference runs `flash_attn_with_kvcache` twice per decoder layer. Self-attention
//! appends each row's new keys and values at the slot's cache length and attends causally
//! over the cache; cross-attention attends over the 1500 encoder positions of the slot,
//! whose keys and values prefill computes once per window. Here self-attention, and
//! cross-attention for prefills and small decode buckets, run on `attn_decode_f16`
//! (`rust/kernels/src/attention.rs`), split over the keys when the batch is too small to
//! fill the GPU, then merged by `attn_combine_f16`.
//!
//! Large decode buckets ([`uses_absorbed`]) run cross-attention over the encoder output
//! `xa` itself instead, with the key projection folded into the query and the value
//! projection applied after the softmax sum (`softmax(q_h W_k,h xa^T) xa W_v,h^T + b_v,h`,
//! exact in real arithmetic since the key has no bias and the weights sum to 1).
//! [`DecoderAttn::cross_absorbed`] runs the middle, `O' = softmax(Q' xa^T) xa` per token
//! and head, on `cross_absorb_f16` (`rust/kernels/src/cross.rs`); the decoder runs the two
//! per-head projections around it ([`CrossKv`]). That reads `xa` (3.84 MB a row) once per
//! layer instead of 7.68 MB of cached keys and values. The caches are still filled for
//! every slot, since a slot moves between buckets as rows finish.
//!
//! # Layout
//!
//! Every KV cache is FP16 `[slots + 1, positions, HEADS, HEAD_DIM]`, token-major like the
//! reference's `FlashKVCache` and like the output of the K/V projection GEMMs, so prefill
//! writes a window's cross keys and values with no transpose: self caches hold
//! [`SELF_CAP`] positions, cross caches [`CROSS_LEN`]. The encoder outputs are FP16
//! `[slots + 1, CROSS_LEN, STATE]`, shared by every layer. Slot `slots` is the trash slot
//! that CUDA-graph padding rows point at (the reference's `--pad-fix`): its length is reset
//! to 0 after every step and its output is never read. All layers share one per-slot
//! length array, since every layer appends the same tokens.
//!
//! # Capture safety
//!
//! Slot indices and lengths live on the device and every call is a fixed sequence of
//! launches on the one stream, with no allocation, synchronization or host read, so a
//! decode step can be captured into a CUDA graph per bucket. The path and the split count
//! depend only on the row count, the tokens per row and the SM count, which a bucket fixes

use eyre::{Result, ensure};

use super::Blas;
use super::blas::{BatchedKey, Strided};
use super::device::{Arg, Buf, Gpu, blocks};

/// Rows from which a decode step (one token a row) runs the absorbed cross-attention
///
/// Per-layer cross-attention time on the RTX 5090 (us, median of 25 with L2 flushed and
/// every row on its own slot, `cross_bench` in `tests/gpu_decoder_attn.rs`): the K/V path
/// is `attn_decode_f16` plus `attn_combine_f16`, the absorbed path the two projection GEMMs
/// around `cross_absorb_f16` and `cross_combine_f16`
///
/// | rows | 1 | 2 | 4 | 8 | 16 | 32 | 42 | 52 | 54 |
/// |------|---|---|---|---|----|----|----|----|----|
/// | K/V | 16.4 | 20.0 | 28.7 | 48.7 | 85.4 | 165.4 | 204.8 | 254.0 | 264.2 |
/// | absorbed | 41.0 | 41.0 | 47.1 | 58.3 | 81.9 | 123.9 | 151.6 | 180.2 | 182.3 |
///
/// The 16-row bucket stays on the K/V path: 4% is not worth the absorbed path's other
/// rounding of the same attention. Below that the absorbed path pays for two GEMMs and a
/// kernel that can't fill the GPU from a few rows. Prefills stay on the K/V path too: at 7
/// tokens a row the absorbed kernel runs 140 query rows against the full `xa` per row and
/// takes twice as long (52 rows: 733 us against 383)
pub const ABSORBED_MIN_ROWS: usize = 32;

/// Whether a batch of `rows` rows of `q_len` tokens runs the absorbed cross-attention
/// ([`DecoderAttn::cross_absorbed`]) rather than the one over the K/V caches
/// ([`DecoderAttn::cross_attn`]); see [`ABSORBED_MIN_ROWS`]
pub fn uses_absorbed(rows: usize, q_len: usize) -> bool {
    q_len == 1 && rows >= ABSORBED_MIN_ROWS
}

/// Frames per tile of `cross_absorb_f16` (`TILE` in `rust/kernels/src/cross.rs`)
const CROSS_TILE: usize = 16;
/// Tiles over the encoder frames
pub const CROSS_TILES: usize = CROSS_LEN.div_ceil(CROSS_TILE);
/// Dynamic shared memory of `cross_absorb_f16` (`SMEM_BYTES` there): two `[16, 1288]` FP16
/// `xa` tiles, 8 partial `[20, 20]` FP32 score tiles, a `[32, 24]` FP16 P tile and three
/// 24-float vectors
const CROSS_SMEM: usize =
    2 * CROSS_TILE * (STATE + 8) * 2 + 8 * HEADS * 20 * 4 + 32 * 24 * 2 + 3 * 24 * 4;

/// Attention heads in each decoder layer
pub const HEADS: usize = 20;
/// Dimensions per head
pub const HEAD_DIM: usize = 64;
/// Decoder width
pub const STATE: usize = HEADS * HEAD_DIM;
/// Self-attention cache positions per slot: the decoder's text context, which the
/// transcriber's `max_tokens` (448) also caps
pub const SELF_CAP: usize = 448;
/// Cross-attention positions: the encoder's audio context
pub const CROSS_LEN: usize = 1500;
/// Softmax scale, `HEAD_DIM^-0.5`
pub const SCALE: f32 = 0.125;

/// Attention blocks the split plan aims for per SM. Decode attention is bound by K/V
/// bandwidth, so the grid needs enough blocks streaming at once; a later timing pass can
/// tune this
const BLOCKS_PER_SM: usize = 4;
/// Keys one attention block covers per unrolled step; splits are multiples of it
const STEP: usize = 64;
/// FP16 NaN, for outputs a test must see overwritten
pub const F16_NAN: u16 = 0x7e00;

/// FP16 token rows in device memory: token `i` starts at `ptr + 2 * i * stride` bytes
///
/// A stride above [`STATE`] reads Q, K or V straight out of a fused projection output
#[derive(Clone, Copy, Debug)]
pub struct Rows {
    pub ptr: u64,
    /// Elements between consecutive tokens
    pub stride: usize,
}

impl Rows {
    /// Dense `[tokens, STATE]` rows
    pub fn dense(ptr: u64) -> Self {
        Self { ptr, stride: STATE }
    }

    /// The kernel slice covering `tokens` tokens of [`STATE`] elements each
    fn arg(&self, tokens: usize) -> Result<Arg> {
        ensure!(
            self.stride >= STATE && self.stride.is_multiple_of(8) && self.ptr.is_multiple_of(16),
            "token rows need a stride of at least {STATE} elements, a multiple of 8, and a \
             16-byte aligned start for 128-bit loads (stride {}, address {:#x})",
            self.stride,
            self.ptr
        );
        Ok(Arg::slice(self.ptr, (tokens - 1) * self.stride + STATE))
    }
}

/// A step's batch: `rows` rows of `q_len` new tokens each, row `i` on the cache slot in
/// `slots[i]`, an `i32` array on the device (so a captured step reads whatever slots the
/// host last wrote there)
#[derive(Clone, Copy, Debug)]
pub struct Batch {
    pub slots: u64,
    pub rows: usize,
    pub q_len: usize,
}

impl Batch {
    fn slots_arg(&self) -> Arg {
        Arg::slice(self.slots, self.rows)
    }

    fn tokens(&self) -> usize {
        self.rows * self.q_len
    }
}

/// How one attention launch splits the keys
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitPlan {
    pub splits: usize,
    /// Keys per split, a multiple of the kernel's 64-key step; the last split also takes
    /// every key past `splits * chunk`
    pub chunk: usize,
}

impl SplitPlan {
    /// The split for `rows * q_len` queries over at most `keys` keys on `sms` SMs: enough
    /// splits that the grid reaches [`BLOCKS_PER_SM`] blocks per SM, but no split under
    /// one 64-key step
    pub fn new(rows: usize, q_len: usize, keys: usize, sms: usize) -> Self {
        let queries = (rows * q_len * HEADS).max(1);
        let want = (BLOCKS_PER_SM * sms).div_ceil(queries);
        let most = keys.div_ceil(STEP).max(1);
        let splits = want.clamp(1, most);
        let chunk = keys.div_ceil(splits).next_multiple_of(STEP);
        Self {
            splits: keys.div_ceil(chunk).max(1),
            chunk,
        }
    }
}

/// One layer's cross-attention key weight, value weight and value bias (device
/// addresses), for the two per-head GEMMs around [`DecoderAttn::cross_absorbed`]
#[derive(Clone, Copy, Debug)]
pub struct CrossKv {
    /// FP16 `[STATE, STATE]`, `nn.Linear` layout (output rows)
    pub wk: u64,
    pub wv: u64,
    /// FP16 `[STATE]`
    pub bv: u64,
}

impl CrossKv {
    /// `qk[tok, h] = q[tok, h] @ W_k,h * SCALE` for every head: the cross query `[tokens,
    /// STATE]` through each head's slice `W_k,h` (`[HEAD_DIM, STATE]`, rows `64h..` of the
    /// key weight) into FP16 `[tokens, HEADS, STATE]`, the softmax scale (a power of two)
    /// applied to the FP32 sum
    pub fn query_keys(&self, blas: &Blas, q: u64, qk: u64, tokens: usize) -> Result<()> {
        let key = BatchedKey {
            // column-major [STATE, HEAD_DIM] views of each head's rows of W_k
            a: Strided {
                rows: STATE,
                cols: HEAD_DIM,
                ld: STATE,
                stride: HEAD_DIM * STATE,
            },
            trans_a: false,
            // the head's 64 query columns of each token
            b: Strided {
                rows: HEAD_DIM,
                cols: tokens,
                ld: STATE,
                stride: HEAD_DIM,
            },
            // token rows of HEADS * STATE, head h at STATE * h
            d: Strided {
                rows: STATE,
                cols: tokens,
                ld: HEADS * STATE,
                stride: STATE,
            },
            batch: HEADS,
            bias_stride: None,
        };
        blas.matmul_batched(key, SCALE, (self.wk, q, qk), 0)
    }

    /// `out[tok, 64h..] = o[tok, h] @ W_v,h^T + b_v,h` for every head: the attention output
    /// `[tokens, HEADS, STATE]` through each head's value rows into FP16 `[tokens, STATE]`,
    /// the bias added in the FP32 epilogue
    pub fn values(&self, blas: &Blas, o: u64, out: u64, tokens: usize) -> Result<()> {
        let key = BatchedKey {
            a: Strided {
                rows: STATE,
                cols: HEAD_DIM,
                ld: STATE,
                stride: HEAD_DIM * STATE,
            },
            trans_a: true,
            b: Strided {
                rows: STATE,
                cols: tokens,
                ld: HEADS * STATE,
                stride: STATE,
            },
            d: Strided {
                rows: HEAD_DIM,
                cols: tokens,
                ld: STATE,
                stride: HEAD_DIM,
            },
            batch: HEADS,
            bias_stride: Some(HEAD_DIM),
        };
        blas.matmul_batched(key, 1.0, (self.wv, o, out), self.bv)
    }
}

/// How one absorbed cross-attention launch splits the encoder frames
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AbsorbedPlan {
    pub splits: usize,
    /// 16-frame tiles per split; the last split takes what is left, at least one
    pub tiles: usize,
}

impl AbsorbedPlan {
    /// The plan closest to `splits` splits that leaves no split empty
    pub fn with_splits(splits: usize) -> Self {
        let tiles = CROSS_TILES.div_ceil(splits.clamp(1, CROSS_TILES));
        Self {
            splits: CROSS_TILES.div_ceil(tiles),
            tiles,
        }
    }

    /// The split for `tokens` tokens on `sms` SMs: one block per SM in one wave, since a
    /// block holds an SM's registers and most of its shared memory, down to one tile a split
    pub fn new(tokens: usize, sms: usize) -> Self {
        Self::with_splits((sms / tokens.max(1)).clamp(1, ABSORBED_MAX_SPLITS))
    }
}

/// Most splits [`AbsorbedPlan::new`] picks: one 16-frame tile each
pub const ABSORBED_MAX_SPLITS: usize = CROSS_TILES;

/// Self-attention KV caches for every decoder layer, the per-slot lengths, and each slot's
/// encoder output, which cross-attention reads in every layer
pub struct KvCaches {
    layers: usize,
    slots: usize,
    self_k: Buf,
    self_v: Buf,
    cross_k: Buf,
    cross_v: Buf,
    /// FP16 `[slots + 1, CROSS_LEN, STATE]`, the reference's `xa_cache`; the trash slot's
    /// stays zero unless a test poisons it
    xa: Buf,
    lens: Buf,
}

/// Bytes of one slot's encoder output
pub const XA_BYTES: usize = CROSS_LEN * STATE * 2;

impl KvCaches {
    /// Zeroed caches for `layers` layers and `slots` slots, plus the trash slot
    pub fn new(gpu: &Gpu, layers: usize, slots: usize) -> Result<Self> {
        let n = slots + 1;
        let self_bytes = layers * n * SELF_CAP * STATE * 2;
        let cross_bytes = layers * n * CROSS_LEN * STATE * 2;
        Ok(Self {
            layers,
            slots,
            self_k: gpu.zeros(self_bytes)?,
            self_v: gpu.zeros(self_bytes)?,
            cross_k: gpu.zeros(cross_bytes)?,
            cross_v: gpu.zeros(cross_bytes)?,
            xa: gpu.zeros(n * XA_BYTES)?,
            lens: gpu.zeros(n * 4)?,
        })
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    /// Real slots, not counting the trash slot
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// The slot padding rows point at
    pub fn trash(&self) -> usize {
        self.slots
    }

    /// Per-slot cache lengths, `i32 [slots + 1]`
    pub fn lens(&self) -> &Buf {
        &self.lens
    }

    /// Device bytes held
    pub fn bytes(&self) -> usize {
        [
            &self.self_k,
            &self.self_v,
            &self.cross_k,
            &self.cross_v,
            &self.xa,
            &self.lens,
        ]
        .iter()
        .map(|b| b.bytes())
        .sum()
    }

    fn layer_bytes(&self, positions: usize) -> usize {
        (self.slots + 1) * positions * STATE * 2
    }

    /// Self-attention K and V of `layer` as kernel slices
    fn self_layer(&self, layer: usize) -> Result<(Arg, Arg)> {
        ensure!(layer < self.layers, "layer {layer} of {}", self.layers);
        let bytes = self.layer_bytes(SELF_CAP);
        let elems = bytes / 2;
        Ok((
            Arg::slice(self.self_k.at(layer * bytes), elems),
            Arg::slice(self.self_v.at(layer * bytes), elems),
        ))
    }

    /// Cross-attention K and V of `layer` as kernel slices
    fn cross_layer(&self, layer: usize) -> Result<(Arg, Arg)> {
        ensure!(layer < self.layers, "layer {layer} of {}", self.layers);
        let bytes = self.layer_bytes(CROSS_LEN);
        let elems = bytes / 2;
        Ok((
            Arg::slice(self.cross_k.at(layer * bytes), elems),
            Arg::slice(self.cross_v.at(layer * bytes), elems),
        ))
    }

    /// Device addresses of the self-attention K and V rows `[SELF_CAP, STATE]` of one slot
    pub fn self_slot(&self, layer: usize, slot: usize) -> Result<(u64, u64)> {
        ensure!(
            layer < self.layers && slot <= self.slots,
            "layer {layer} slot {slot}"
        );
        let at = layer * self.layer_bytes(SELF_CAP) + slot * SELF_CAP * STATE * 2;
        Ok((self.self_k.at(at), self.self_v.at(at)))
    }

    /// Device addresses of the cross-attention K and V rows `[CROSS_LEN, STATE]` of one slot
    pub fn cross_slot(&self, layer: usize, slot: usize) -> Result<(u64, u64)> {
        ensure!(
            layer < self.layers && slot <= self.slots,
            "layer {layer} slot {slot}"
        );
        let at = layer * self.layer_bytes(CROSS_LEN) + slot * CROSS_LEN * STATE * 2;
        Ok((self.cross_k.at(at), self.cross_v.at(at)))
    }

    /// Computes the cross keys and values of `layer` for the windows of a prefill batch
    /// and stores window `i` in slot `slots[i]`
    ///
    /// `xa` is the encoder output `[windows * CROSS_LEN, STATE]`. As the reference runs
    /// `key(xa)` and `value(xa)` on the whole batch, the GEMMs take every window at once
    /// (torch's row count, so cuBLAS picks the same kernel): `K = xa @ wk^T` (no bias) and
    /// `V = xa @ wv^T + bv`. Consecutive ascending slots get the GEMM output directly;
    /// otherwise it lands in `scratch` (at least `windows * CROSS_LEN * STATE * 2` bytes)
    /// and is copied per slot. Prefill knows its slots on the host, so this takes them
    /// from there
    pub fn fill_cross(
        &self,
        gpu: &Gpu,
        blas: &Blas,
        layer: usize,
        xa: u64,
        (wk, wv, bv): (u64, u64, u64),
        (slots, scratch): (&[usize], Option<&Buf>),
    ) -> Result<()> {
        let windows = slots.len();
        ensure!(windows > 0, "fill_cross with no windows");
        ensure!(
            slots.iter().all(|&s| s < self.slots),
            "fill_cross slots {slots:?} outside the {} real slots",
            self.slots
        );
        let rows = windows * CROSS_LEN;
        let slot_bytes = CROSS_LEN * STATE * 2;
        let (k0, v0) = self.cross_slot(layer, slots[0])?;
        let consecutive = slots.windows(2).all(|w| w[1] == w[0] + 1);
        if consecutive {
            blas.linear(xa, wk, k0, (rows, STATE, STATE))?;
            return blas.linear_bias(xa, wv, bv, v0, (rows, STATE, STATE));
        }

        let scratch = scratch.ok_or_else(|| {
            eyre::eyre!("fill_cross into non-consecutive slots {slots:?} needs a scratch buffer")
        })?;
        ensure!(
            scratch.bytes() >= windows * slot_bytes,
            "fill_cross scratch of {} bytes, need {}",
            scratch.bytes(),
            windows * slot_bytes
        );
        blas.linear(xa, wk, scratch.ptr(), (rows, STATE, STATE))?;
        for (i, &s) in slots.iter().enumerate() {
            gpu.copy_dtod(
                scratch.at(i * slot_bytes),
                self.cross_slot(layer, s)?.0,
                slot_bytes,
            )?;
        }
        blas.linear_bias(xa, wv, bv, scratch.ptr(), (rows, STATE, STATE))?;
        for (i, &s) in slots.iter().enumerate() {
            gpu.copy_dtod(
                scratch.at(i * slot_bytes),
                self.cross_slot(layer, s)?.1,
                slot_bytes,
            )?;
        }

        Ok(())
    }

    /// Device address of slot `slot`'s encoder output `[CROSS_LEN, STATE]`, the trash slot
    /// included
    pub fn xa_slot(&self, slot: usize) -> Result<u64> {
        ensure!(slot <= self.slots, "encoder output of slot {slot}");
        Ok(self.xa.at(slot * XA_BYTES))
    }

    /// Copies window `i` of the encoder output `xa` (`[windows * CROSS_LEN, STATE]` FP16)
    /// into slot `slots[i]`, for the DTW forward and the absorbed cross-attention
    pub fn store_xa(&self, gpu: &Gpu, xa: u64, slots: &[usize]) -> Result<()> {
        ensure!(
            slots.iter().all(|&s| s < self.slots),
            "encoder outputs for slots {slots:?} outside the {} real slots",
            self.slots
        );
        for (i, &s) in slots.iter().enumerate() {
            gpu.copy_dtod(xa + (i * XA_BYTES) as u64, self.xa_slot(s)?, XA_BYTES)?;
        }

        Ok(())
    }

    /// Sets the cache length of each row's slot to 0, before a prefill's layers run
    pub fn reset_lens(&self, gpu: &Gpu, batch: Batch) -> Result<()> {
        self.update_lens(gpu, batch, 0, 0)
    }

    /// Adds `q_len` to the cache length of each row's slot after a step's layers have run,
    /// and resets the trash slot to 0
    pub fn advance_lens(&self, gpu: &Gpu, batch: Batch) -> Result<()> {
        self.update_lens(gpu, batch, 1, batch.q_len)
    }

    fn update_lens(&self, gpu: &Gpu, batch: Batch, keep: u32, add: usize) -> Result<()> {
        gpu.launch(
            "kv_lens",
            (blocks(batch.rows, 64), 1, 1),
            (64, 1, 1),
            0,
            &[
                self.lens.slice::<i32>(),
                batch.slots_arg(),
                keep.into(),
                add.into(),
                self.trash().into(),
            ],
        )
    }
}

/// The attention launches and their split workspaces
pub struct DecoderAttn {
    sms: usize,
    part_o: Buf,
    part_lse: Buf,
    /// Partial results the workspace holds, `(row, token, head, split)` entries
    parts: usize,
    /// The split cross-attention's partials, FP32 `[tokens, splits, HEADS, STATE]`, and
    /// their log2-sum-exps `[tokens, splits, HEADS]`
    cross_part_o: Buf,
    cross_part_lse: Buf,
    /// (token, split) pairs the cross partials hold
    cross_parts: usize,
}

impl DecoderAttn {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        ensure!(
            gpu.max_shared_per_block() >= CROSS_SMEM,
            "cross-attention needs {CROSS_SMEM} bytes of shared memory per block; this GPU \
             allows {}",
            gpu.max_shared_per_block()
        );
        // a plan splits only while rows * q_len * HEADS is under BLOCKS_PER_SM * sms, and
        // then makes fewer than BLOCKS_PER_SM * sms + rows * q_len * HEADS partials
        let parts = 2 * BLOCKS_PER_SM * gpu.sms();
        // a split cross plan keeps tokens * splits at most sms, or ABSORBED_MAX_SPLITS when
        // a single token gets more splits than that would allow
        let cross_parts = gpu.sms().max(ABSORBED_MAX_SPLITS);
        Ok(Self {
            sms: gpu.sms(),
            part_o: gpu.alloc(parts * HEAD_DIM * 4)?,
            part_lse: gpu.alloc(parts * 4)?,
            parts,
            cross_part_o: gpu.alloc(cross_parts * HEADS * STATE * 4)?,
            cross_part_lse: gpu.alloc(cross_parts * HEADS * 4)?,
            cross_parts,
        })
    }

    /// The split the next self-attention launch for this batch shape uses
    pub fn self_plan(&self, rows: usize, q_len: usize) -> SplitPlan {
        SplitPlan::new(rows, q_len, SELF_CAP, self.sms)
    }

    /// The split the next cross-attention launch over the K/V caches for this batch shape
    /// uses
    pub fn cross_plan(&self, rows: usize, q_len: usize) -> SplitPlan {
        SplitPlan::new(rows, q_len, CROSS_LEN, self.sms)
    }

    /// The frame split the next absorbed cross-attention launch for this batch shape uses
    pub fn absorbed_plan(&self, rows: usize, q_len: usize) -> AbsorbedPlan {
        AbsorbedPlan::new(rows * q_len, self.sms)
    }

    /// Self-attention of `layer` for the batch's new tokens
    ///
    /// Appends `k` and `v` (token `row * q_len + t`) to each row's slot at its cache
    /// length plus `t`, and writes FP16 `out[rows * q_len, STATE]`: query `t` attends to
    /// the slot's cached keys and new keys `0..=t`, as `flash_attn_with_kvcache` with
    /// `causal=True` does. Lengths are not advanced here: every layer appends at the same
    /// position, so the caller runs [`KvCaches::advance_lens`] once after the last layer
    pub fn self_attn(
        &self,
        gpu: &Gpu,
        caches: &KvCaches,
        layer: usize,
        batch: Batch,
        (q, k, v): (Rows, Rows, Rows),
        out: u64,
    ) -> Result<()> {
        ensure!(
            k.stride == v.stride,
            "new K and V rows need one stride ({} and {})",
            k.stride,
            v.stride
        );
        let (k_cache, v_cache) = caches.self_layer(layer)?;
        let plan = self.self_plan(batch.rows, batch.q_len);
        self.launch(
            gpu,
            (plan, SELF_CAP),
            batch,
            (k_cache, v_cache, caches.lens.slice::<i32>()),
            (q, Some((k, v))),
            out,
        )
    }

    /// Cross-attention of `layer`: every query of each row attends to all [`CROSS_LEN`]
    /// keys of its slot, read only, writing FP16 `out[rows * q_len, STATE]`
    pub fn cross_attn(
        &self,
        gpu: &Gpu,
        caches: &KvCaches,
        layer: usize,
        batch: Batch,
        q: Rows,
        out: u64,
    ) -> Result<()> {
        let (k_cache, v_cache) = caches.cross_layer(layer)?;
        let plan = self.cross_plan(batch.rows, batch.q_len);
        self.launch(
            gpu,
            (plan, CROSS_LEN),
            batch,
            (k_cache, v_cache, Arg::EMPTY),
            (q, None),
            out,
        )
    }

    /// The middle of the absorbed cross-attention: for each token, `O' = softmax(Q' xa^T)
    /// xa` with its 20 heads as the query rows and `xa` its row's slot's encoder output.
    /// `q` is FP16 `Q'` `[rows * q_len, HEADS, STATE]` (each head's query through its key
    /// projection, the softmax scale folded in), `out` FP16 `O'` in the same layout
    pub fn cross_absorbed(
        &self,
        gpu: &Gpu,
        caches: &KvCaches,
        batch: Batch,
        q: u64,
        out: u64,
    ) -> Result<()> {
        self.cross_absorbed_with(
            gpu,
            caches,
            batch,
            (q, out),
            self.absorbed_plan(batch.rows, batch.q_len),
        )
    }

    /// [`DecoderAttn::cross_absorbed`] on an explicit frame split
    pub fn cross_absorbed_with(
        &self,
        gpu: &Gpu,
        caches: &KvCaches,
        batch: Batch,
        (q, out): (u64, u64),
        plan: AbsorbedPlan,
    ) -> Result<()> {
        let tokens = batch.tokens();
        ensure!(tokens > 0, "cross-attention over no tokens");
        ensure!(
            q.is_multiple_of(16) && out.is_multiple_of(16),
            "cross-attention query at {q:#x} or output at {out:#x} misaligned"
        );
        ensure!(
            plan.splits >= 1
                && plan.tiles >= 1
                && (plan.splits - 1) * plan.tiles < CROSS_TILES
                && plan.splits * plan.tiles >= CROSS_TILES,
            "cross plan {plan:?} leaves frames out or a split empty"
        );
        let split = plan.splits > 1;
        let parts = tokens * plan.splits;
        if split {
            ensure!(
                parts <= self.cross_parts,
                "cross plan {plan:?} for {tokens} tokens overflows the {} partials",
                self.cross_parts
            );
        }

        let rows_elems = tokens * HEADS * STATE;
        let out_arg = Arg::slice(out, rows_elems);
        let (out_k, part_o, part_lse) = if split {
            (
                Arg::EMPTY,
                Arg::slice(self.cross_part_o.ptr(), parts * HEADS * STATE),
                Arg::slice(self.cross_part_lse.ptr(), parts * HEADS),
            )
        } else {
            (out_arg, Arg::EMPTY, Arg::EMPTY)
        };
        gpu.launch(
            "cross_absorb_f16",
            (plan.splits as u32, u32::try_from(tokens)?, 1),
            (256, 1, 1),
            CROSS_SMEM as u32,
            &[
                out_k,
                part_o,
                part_lse,
                caches.xa.slice::<u16>(),
                Arg::slice(q, rows_elems),
                batch.slots_arg(),
                batch.q_len.into(),
                plan.splits.into(),
                plan.tiles.into(),
            ],
        )?;

        if split {
            gpu.launch(
                "cross_combine_f16",
                (u32::try_from(tokens * HEADS)?, (STATE / 256) as u32, 1),
                (64, 1, 1),
                0,
                &[
                    out_arg,
                    Arg::slice(self.cross_part_o.ptr(), parts * HEADS * STATE),
                    Arg::slice(self.cross_part_lse.ptr(), parts * HEADS),
                    plan.splits.into(),
                ],
            )?;
        }

        Ok(())
    }

    fn launch(
        &self,
        gpu: &Gpu,
        (plan, cap): (SplitPlan, usize),
        batch: Batch,
        (k_cache, v_cache, lens): (Arg, Arg, Arg),
        (q, new): (Rows, Option<(Rows, Rows)>),
        out: u64,
    ) -> Result<()> {
        ensure!(batch.tokens() > 0, "attention over no tokens");
        let tokens = batch.tokens();
        ensure!(
            out.is_multiple_of(4),
            "attention output at {out:#x} is misaligned"
        );
        let queries = tokens * HEADS;
        let split = plan.splits > 1;
        if split {
            ensure!(
                queries * plan.splits <= self.parts,
                "split plan {plan:?} for {queries} queries overflows the {} partials",
                self.parts
            );
        }

        let (k_new, v_new, kv_stride) = match new {
            Some((k, v)) => (k.arg(tokens)?, v.arg(tokens)?, k.stride),
            None => (Arg::EMPTY, Arg::EMPTY, 0),
        };
        let out_arg = Arg::slice(out, tokens * STATE);
        let (out_k, part_o, part_lse) = if split {
            (
                Arg::EMPTY,
                Arg::slice(self.part_o.ptr(), queries * plan.splits * HEAD_DIM),
                Arg::slice(self.part_lse.ptr(), queries * plan.splits),
            )
        } else {
            (out_arg, Arg::EMPTY, Arg::EMPTY)
        };
        let scale_log2 = SCALE * std::f32::consts::LOG2_E;
        gpu.launch(
            "attn_decode_f16",
            (u32::try_from(queries)?, plan.splits as u32, 1),
            (128, 1, 1),
            0,
            &[
                out_k,
                part_o,
                part_lse,
                k_cache,
                v_cache,
                q.arg(tokens)?,
                k_new,
                v_new,
                lens,
                batch.slots_arg(),
                batch.q_len.into(),
                q.stride.into(),
                kv_stride.into(),
                cap.into(),
                plan.splits.into(),
                plan.chunk.into(),
                scale_log2.into(),
            ],
        )?;

        if split {
            gpu.launch(
                "attn_combine_f16",
                (u32::try_from(queries)?, 1, 1),
                (HEAD_DIM as u32, 1, 1),
                0,
                &[
                    out_arg,
                    Arg::slice(self.part_o.ptr(), queries * plan.splits * HEAD_DIM),
                    Arg::slice(self.part_lse.ptr(), queries * plan.splits),
                    plan.splits.into(),
                ],
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{CROSS_LEN, HEADS, SELF_CAP, STEP, SplitPlan};

    /// Every plan covers all keys, keeps the grid near the target and never overflows
    /// the workspace `DecoderAttn::new` sizes
    #[test]
    fn split_plans_cover_the_keys() {
        for sms in [1, 84, 170] {
            for q_len in [1, 7] {
                for rows in [1, 2, 4, 8, 16, 32, 42, 52, 54] {
                    for keys in [SELF_CAP, CROSS_LEN] {
                        let p = SplitPlan::new(rows, q_len, keys, sms);
                        assert!(p.chunk.is_multiple_of(STEP), "{p:?}");
                        assert!(p.splits * p.chunk >= keys, "{p:?} misses keys");
                        assert!((p.splits - 1) * p.chunk < keys, "{p:?} has an empty split");
                        let queries = rows * q_len * HEADS;
                        if p.splits > 1 {
                            assert!(queries * p.splits <= 8 * sms, "{p:?} rows {rows}");
                        }
                    }
                }
            }
        }
    }
}
