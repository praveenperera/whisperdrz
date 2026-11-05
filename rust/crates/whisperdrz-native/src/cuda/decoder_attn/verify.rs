//! Attention launches of speculative verify steps: packed rows of 1 to
//! [`MAX_QUERIES`] tokens on `attn_verify_q*_f16` (`rust/kernels/src/attention_varlen.rs`)
//!
//! A verify row attends like a decode row of its own query count: causal self-attention
//! over its slot's cached keys plus its new ones, appended at its cache length, and
//! cross-attention over the slot's 1500 encoder keys. One block serves every query of a
//! (row, head, split), so a row's keys and values stream once whatever its query count.
//! The split plan is the one-token step's for the row count, so a step of one-token rows
//! runs the same blocks, partials and merge as [`DecoderAttn`](super::DecoderAttn) and
//! gives the same bits

use eyre::{Result, ensure};

use super::{CROSS_LEN, HEAD_DIM, HEADS, KvCaches, Rows, SCALE, SELF_CAP, STATE, SplitPlan};
use crate::cuda::device::{Arg, Buf, Gpu};

/// Most tokens a verify row feeds: the widest kernel variant
pub const MAX_QUERIES: usize = 8;

/// A verify step's packed rows, `i32` device arrays: row `r`'s slot, its `rows + 1`
/// token offsets, and its cache length before the step
#[derive(Clone, Copy, Debug)]
pub struct Ragged {
    pub slots: u64,
    pub starts: u64,
    pub lens: u64,
    pub rows: usize,
    pub tokens: usize,
    /// The most tokens one row holds
    pub max_q: usize,
}

impl Ragged {
    /// The kernel variant for the widest row
    fn kernel(&self) -> Result<&'static str> {
        Ok(match self.max_q {
            1..=2 => "attn_verify_q2_f16",
            3..=4 => "attn_verify_q4_f16",
            5..=MAX_QUERIES => "attn_verify_q8_f16",
            q => eyre::bail!("verify row of {q} tokens; at most {MAX_QUERIES}"),
        })
    }
}

/// The verify attention launches and their split workspace
pub struct VerifyAttn {
    sms: usize,
    part_o: Buf,
    part_lse: Buf,
    /// `(token, head, split)` partials the workspace holds
    parts: usize,
}

impl VerifyAttn {
    /// A split workspace for steps of up to `max_tokens` tokens, one row per token at
    /// most
    pub fn new(gpu: &Gpu, max_tokens: usize) -> Result<Self> {
        let sms = gpu.sms();
        // the widest plan over any row count: a row holds at most MAX_QUERIES tokens
        let parts = (1..=max_tokens)
            .map(|rows| {
                let splits = SplitPlan::new(rows, 1, CROSS_LEN, sms)
                    .splits
                    .max(SplitPlan::new(rows, 1, SELF_CAP, sms).splits);
                max_tokens.min(rows * MAX_QUERIES) * HEADS * splits
            })
            .max()
            .unwrap_or(0);
        Ok(Self {
            sms,
            part_o: gpu.alloc(parts.max(1) * HEAD_DIM * 4)?,
            part_lse: gpu.alloc(parts.max(1) * 4)?,
            parts,
        })
    }

    /// Device bytes held
    pub fn bytes(&self) -> usize {
        self.part_o.bytes() + self.part_lse.bytes()
    }

    /// The split the self-attention of a step of `rows` rows uses
    pub fn self_plan(&self, rows: usize) -> SplitPlan {
        SplitPlan::new(rows, 1, SELF_CAP, self.sms)
    }

    /// The split the cross-attention of a step of `rows` rows uses
    pub fn cross_plan(&self, rows: usize) -> SplitPlan {
        SplitPlan::new(rows, 1, CROSS_LEN, self.sms)
    }

    /// Self-attention of `layer`: appends `k` and `v` to each row's slot at its cache
    /// length plus the token's index in the row, and writes FP16 `out[tokens, STATE]`
    pub fn self_attn(
        &self,
        gpu: &Gpu,
        caches: &KvCaches,
        layer: usize,
        rows: Ragged,
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
        let plan = self.self_plan(rows.rows);
        self.launch(
            gpu,
            (plan, SELF_CAP),
            rows,
            (k_cache, v_cache, Arg::slice(rows.lens, rows.rows)),
            (q, Some((k, v))),
            out,
        )
    }

    /// Cross-attention of `layer`: every query attends to all [`CROSS_LEN`] keys of its
    /// row's slot, read only
    pub fn cross_attn(
        &self,
        gpu: &Gpu,
        caches: &KvCaches,
        layer: usize,
        rows: Ragged,
        q: Rows,
        out: u64,
    ) -> Result<()> {
        let (k_cache, v_cache) = caches.cross_layer(layer)?;
        let plan = self.cross_plan(rows.rows);
        self.launch(
            gpu,
            (plan, CROSS_LEN),
            rows,
            (k_cache, v_cache, Arg::EMPTY),
            (q, None),
            out,
        )
    }

    fn launch(
        &self,
        gpu: &Gpu,
        (plan, cap): (SplitPlan, usize),
        rows: Ragged,
        (k_cache, v_cache, lens): (Arg, Arg, Arg),
        (q, new): (Rows, Option<(Rows, Rows)>),
        out: u64,
    ) -> Result<()> {
        let tokens = rows.tokens;
        ensure!(
            rows.rows > 0 && tokens >= rows.rows,
            "verify attention over {} rows of {tokens} tokens",
            rows.rows
        );
        ensure!(
            out.is_multiple_of(4),
            "attention output at {out:#x} is misaligned"
        );
        let kernel = rows.kernel()?;
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
        let parts = queries * plan.splits;
        let (out_k, part_o, part_lse) = if split {
            (
                Arg::EMPTY,
                Arg::slice(self.part_o.ptr(), parts * HEAD_DIM),
                Arg::slice(self.part_lse.ptr(), parts),
            )
        } else {
            (out_arg, Arg::EMPTY, Arg::EMPTY)
        };
        let scale_log2 = SCALE * std::f32::consts::LOG2_E;
        gpu.launch(
            kernel,
            (u32::try_from(rows.rows * HEADS)?, plan.splits as u32, 1),
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
                Arg::slice(rows.starts, rows.rows + 1),
                lens,
                Arg::slice(rows.slots, rows.rows),
                q.stride.into(),
                kv_stride.into(),
                cap.into(),
                plan.splits.into(),
                plan.chunk.into(),
                scale_log2.into(),
            ],
        )?;

        if split {
            // the partials sit at (token * HEADS + head) * splits + split, the layout the
            // one-token step's merge reads
            gpu.launch(
                "attn_combine_f16",
                (u32::try_from(queries)?, 1, 1),
                (HEAD_DIM as u32, 1, 1),
                0,
                &[
                    out_arg,
                    Arg::slice(self.part_o.ptr(), parts * HEAD_DIM),
                    Arg::slice(self.part_lse.ptr(), parts),
                    plan.splits.into(),
                ],
            )?;
        }

        Ok(())
    }
}
