//! The cuda-oxide small-M linears (`linear_n8_*` in `rust/kernels/src/linear.rs`): `y = x @
//! w^T + b` for the decode step's buckets of up to 16 rows, every weight byte requested at
//! launch, FP32 sums and one FP16 rounding after the bias, optionally GELU after it
//!
//! At 2 to 16 rows torch's call (and every non-split-K cuBLASLt algorithm the heuristic
//! offers) runs the 1280x1280 projections on 80 blocks of 32 threads at about a quarter of
//! DRAM rate, and fc2 at 8 and 16 rows at half the rate it reaches at 4. cuBLASLt's split-K
//! algorithms would stream faster but round as if the product were FP16 before the bias.
//! [`route`] sends the decode cells where these kernels measured at least 1.1x faster than
//! the current pick to them, and everything else stays on [`Blas::linear_bias_decode`]
//!
//! A [`SmallCfg`] names the kernel (row capacity and depth chunks per warp) and its grid
//! (warps along the depth inside a block, depth splits across blocks). Split launches write
//! FP32 partials to a workspace and the last block of each feature tile sums them in a fixed
//! order, with one zeroed counter word per tile that it leaves zeroed again, so launches on
//! one stream can share both buffers and the output is the same run to run. Two streams
//! must not share one [`SmallM`]: a tile's counter is only back at zero once its launch ends
//!
//! [`Blas::linear_bias_decode`]: super::Blas::linear_bias_decode

use eyre::{Result, ensure, eyre};

use super::device::{Arg, Buf, Gpu};

/// Threads per block (`THREADS` in the kernel)
const THREADS: u32 = 128;
const WARPS: usize = 4;
/// Features of one warp tile
const WARP_FEATURES: usize = 8;

/// One kernel and grid for a linear shape
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SmallCfg {
    /// Up to 16 rows when true, else up to 8
    pub r16: bool,
    /// 32-deep chunks of the depth each warp holds
    pub ch: usize,
    /// Warps along the depth inside a block (1, 2 or 4); the other `4 / wk` warps take 8
    /// features each
    pub wk: usize,
    /// Blocks along the depth
    pub splits: usize,
}

impl SmallCfg {
    /// Rows the kernel serves
    pub fn max_rows(&self) -> usize {
        if self.r16 { 16 } else { 8 }
    }

    /// Features per block
    pub fn features(&self) -> usize {
        WARP_FEATURES * (WARPS / self.wk)
    }

    /// Depth per block
    pub fn depth(&self) -> usize {
        self.wk * self.ch * 32
    }

    /// The kernel for this configuration, if it is one of the built variants
    pub fn kernel(&self) -> Option<&'static str> {
        match (self.r16, self.ch) {
            (false, 5) => Some("linear_n8_r8_c5"),
            (true, 10) => Some("linear_n8_r16_c10"),
            _ => None,
        }
    }

    /// Dynamic shared memory per block: the cross-warp reduction area and a flag
    pub fn smem(&self) -> usize {
        (WARPS - 1) * 32 * 16 + 16
    }

    /// FP32 partials a launch of `batch` linears with `n` outputs writes
    pub fn part_floats(&self, n: usize, batch: usize) -> usize {
        if self.splits == 1 {
            0
        } else {
            batch * (n / self.features()) * self.splits * (WARPS / self.wk) * 128
        }
    }

    /// Counter words a launch of `batch` linears with `n` outputs uses
    pub fn counter_words(&self, n: usize, batch: usize) -> usize {
        if self.splits == 1 {
            0
        } else {
            batch * (n / self.features())
        }
    }
}

/// A decode cell routed to the narrow kernels: `rows` rows of an `n_in -> n_out` linear
struct Route {
    rows: usize,
    n_in: usize,
    n_out: usize,
    cfg: SmallCfg,
}

const fn route_to(
    rows: usize,
    n_in: usize,
    n_out: usize,
    r16: bool,
    ch: usize,
    wk: usize,
) -> Route {
    Route {
        rows,
        n_in,
        n_out,
        cfg: SmallCfg {
            r16,
            ch,
            wk,
            splits: n_in / (wk * ch * 32),
        },
    }
}

/// The decode cells these kernels run, from gpu0 job 0c0b14c4 (RTX 5090, weights rotated
/// past L2 as the decode step streams them): each beat the current pick by 1.1x or more,
/// with its per-launch times (current pick -> this kernel) in the comment. fc1 is timed with
/// its GELU, separate for the current pick and fused here. Every other cell, including 1
/// row, keeps the current pick. Measured only on the RTX 5090, so other GPUs keep it too
const ROUTES: &[Route] = &[
    route_to(2, 1280, 1280, false, 5, 2),  // 7.39 -> 4.74 us
    route_to(4, 1280, 1280, false, 5, 2),  // 7.38 -> 4.72 us
    route_to(8, 1280, 1280, false, 5, 2),  // 7.39 -> 4.74 us
    route_to(16, 1280, 1280, true, 10, 4), // 7.41 -> 4.51 us
    route_to(16, 1280, 3840, true, 10, 4), // 9.04 -> 7.89 us
    route_to(16, 1280, 5120, true, 10, 4), // 11.55 -> 10.08 us, with GELU
    route_to(8, 5120, 1280, false, 5, 2),  // 21.95 -> 10.95 us
    route_to(16, 5120, 1280, true, 10, 2), // 21.97 -> 11.26 us
];

/// Whether `gpu` is the device [`ROUTES`] was measured on, an RTX 5090 (compute 12.0, 170
/// SMs), the same check as the cuBLASLt decode table's
pub fn measured_on(gpu: &Gpu) -> bool {
    gpu.compute_capability() == (12, 0) && gpu.sms() == 170
}

/// The configuration [`ROUTES`] gives `rows` rows of an `n_in -> n_out` linear, if any
pub fn route((rows, n_in, n_out): (usize, usize, usize)) -> Option<SmallCfg> {
    ROUTES
        .iter()
        .find(|r| r.rows == rows && r.n_in == n_in && r.n_out == n_out)
        .map(|r| r.cfg)
}

/// Element strides of a batched small-M linear: row strides of `x` and `y`, and the offsets
/// of `x`, `w`, `b` and `y` from one batch entry to the next
#[derive(Clone, Copy, Debug)]
pub struct SmallStrides {
    pub ldx: usize,
    pub ldy: usize,
    pub batch: usize,
    pub xz: usize,
    pub wz: usize,
    pub bz: usize,
    pub yz: usize,
}

impl SmallStrides {
    /// One dense linear: `x` `[rows, k]`, `y` `[rows, n]`
    pub fn dense(k: usize, n: usize) -> Self {
        Self {
            ldx: k,
            ldy: n,
            batch: 1,
            xz: 0,
            wz: 0,
            bz: 0,
            yz: 0,
        }
    }
}

/// The split-K workspace and counters the small-M linears share on one stream
pub struct SmallM {
    part: Buf,
    part_floats: usize,
    counters: Buf,
    counter_words: usize,
    /// Whether [`ROUTES`] applies on this GPU
    routed: bool,
}

impl SmallM {
    /// Workspace for launches of up to `part_floats` partials and `counter_words` counters
    pub fn new(gpu: &Gpu, part_floats: usize, counter_words: usize) -> Result<Self> {
        Ok(Self {
            part: gpu.alloc(part_floats.max(4) * 4)?,
            part_floats,
            counters: gpu.zeros(counter_words.max(1) * 4)?,
            counter_words,
            routed: measured_on(gpu),
        })
    }

    /// Workspace for every cell in [`ROUTES`]
    pub fn for_routes(gpu: &Gpu) -> Result<Self> {
        let part = ROUTES.iter().map(|r| r.cfg.part_floats(r.n_out, 1)).max();
        let counters = ROUTES.iter().map(|r| r.cfg.counter_words(r.n_out, 1)).max();
        Self::new(gpu, part.unwrap_or(0), counters.unwrap_or(0))
    }

    /// The routed configuration for a dense decode linear on this GPU, if any: in
    /// [`ROUTES`], measured on this device, and every operand aligned as the kernel loads
    /// it (16 bytes for `x` and `w`, 4 for `y`)
    pub fn route(
        &self,
        (x, w, y): (u64, u64, u64),
        dims: (usize, usize, usize),
    ) -> Option<SmallCfg> {
        let aligned = x.is_multiple_of(16) && w.is_multiple_of(16) && y.is_multiple_of(4);
        route(dims).filter(|_| self.routed && aligned)
    }

    /// `y[r, f] = x[r, :] . w[f, :] + b[f]` for `rows` rows, `k` deep, `n` features, then
    /// GELU when `gelu`, on `cfg`, with `strides` (batched along the grid's z). Capture safe:
    /// it enqueues one launch and allocates nothing
    #[allow(clippy::too_many_arguments)]
    pub fn linear(
        &self,
        gpu: &Gpu,
        cfg: SmallCfg,
        (x, w, b, y): (u64, u64, u64, u64),
        (rows, k, n): (usize, usize, usize),
        s: SmallStrides,
        gelu: bool,
    ) -> Result<()> {
        let name = cfg
            .kernel()
            .ok_or_else(|| eyre!("no small-M kernel for {cfg:?}"))?;
        ensure!(
            rows >= 1
                && rows <= cfg.max_rows()
                && n.is_multiple_of(cfg.features())
                && k == cfg.splits * cfg.depth()
                && [1, 2, 4].contains(&cfg.wk)
                && s.batch >= 1
                && s.ldy.is_multiple_of(2)
                && s.yz.is_multiple_of(2),
            "small-M linear {rows}x{k} -> {n} on {cfg:?}"
        );
        ensure!(
            cfg.part_floats(n, s.batch) <= self.part_floats
                && cfg.counter_words(n, s.batch) <= self.counter_words,
            "small-M workspace too small for {cfg:?} at {n} outputs"
        );
        let last = s.batch - 1;
        let x_len = last * s.xz + (rows - 1) * s.ldx + k;
        let y_len = last * s.yz + (rows - 1) * s.ldy + n;
        let u = |v: usize| u32::try_from(v).map_err(|_| eyre!("small-M size {v} overflows u32"));
        let grid = (u(n / cfg.features())?, u(cfg.splits)?, u(s.batch)?);
        gpu.launch(
            name,
            grid,
            (THREADS, 1, 1),
            u(cfg.smem())?,
            &[
                Arg::slice(y, y_len),
                Arg::slice(self.part.ptr(), self.part_floats),
                Arg::slice(self.counters.ptr(), self.counter_words),
                Arg::slice(x, x_len),
                Arg::slice(w, last * s.wz + n * k),
                Arg::slice(b, last * s.bz + n),
                u(rows)?.into(),
                u(k)?.into(),
                u(n)?.into(),
                u(s.ldx)?.into(),
                u(s.ldy)?.into(),
                u(cfg.wk)?.into(),
                u(cfg.splits)?.into(),
                u32::from(gelu).into(),
                u(s.xz)?.into(),
                u(s.wz)?.into(),
                u(s.bz)?.into(),
                u(s.yz)?.into(),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{ROUTES, route};

    #[test]
    fn every_route_is_a_built_kernel_that_tiles_its_shape() {
        for r in ROUTES {
            let c = r.cfg;
            assert!(c.kernel().is_some(), "{c:?} has no kernel");
            assert!(r.rows <= c.max_rows(), "{} rows on {c:?}", r.rows);
            assert_eq!(c.splits * c.depth(), r.n_in, "{c:?} depth");
            assert!(r.n_out.is_multiple_of(c.features()), "{c:?} features");
            assert_eq!(route((r.rows, r.n_in, r.n_out)), Some(c));
        }
        assert_eq!(route((1, 1280, 1280)), None);
    }
}
