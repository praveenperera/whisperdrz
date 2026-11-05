//! Decode-step linears on cuBLASLt algorithms chosen by measurement
//!
//! torch's `addmm` call asks the heuristic for one algorithm with a 1 MiB workspace. At
//! the decode step's row counts (the graph buckets, 1 to 54 rows) that picks kernels
//! that launch too few blocks to stream the weights at DRAM rate: 80 blocks of 32 threads
//! for the 1280x1280 projections, one 160-block wave of 256x128 tiles for the packed
//! QKV at 32 rows and more. [`Blas::linear_bias_decode`] runs the shapes in [`TABLE`] on
//! the algorithm measured fastest for them on an RTX 5090 (`tests/gpu_gemm_probe.rs`,
//! weights rotated past L2 as the decode step streams them) and every other shape as
//! torch does. The compute type, scale type and bias epilogue are torch's; only the
//! kernel and its split of the inner dimension differ, so outputs differ from torch's
//! in FP16 rounding. A fixed table rather than timing at load keeps the algorithm, and
//! so every output bit, the same from run to run. Split-K entries use the compute-type
//! reduction, FP32 partials summed in a fixed order by a separate kernel

use std::collections::hash_map::Entry;
use std::ffi::c_void;

use cudarc::cublaslt::sys as lt;
use eyre::{Result, bail, ensure};

use super::super::device::Gpu;
use super::{Blas, LinearKey, LinearPlan, LtFns, alignment, check_lt};

/// The decode linears' cuBLASLt workspace, enough for every [`TABLE`] entry's split-K
/// partials (checked when a plan is made; the largest, 1280x1280 at 32 rows split 20
/// ways, needs 3.1 MiB)
pub(super) const DECODE_WORKSPACE: usize = 4 << 20;

/// One cuBLASLt algorithm: the `CUBLASLT_ALGO_CONFIG_*` values that identify it
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LtAlgo {
    pub id: i32,
    pub tile: u32,
    pub stages: u32,
    pub splitk: u32,
    pub reduction: u32,
    pub swizzle: u32,
    pub custom: u32,
    pub inner: u16,
    pub cluster: u16,
}

/// An algorithm the library accepts for a linear shape, with the workspace it needs
#[derive(Clone, Copy, Debug)]
pub struct LtCandidate {
    pub algo: LtAlgo,
    pub workspace: usize,
    /// The heuristic's estimated waves, 0 for split-K variants it didn't return
    pub waves: f32,
    /// Whether the heuristic returned it, rather than the probe's split-K sweep
    pub heuristic: bool,
}

/// A measured algorithm for `rows` rows of an `n_in -> n_out` linear with bias
struct Pick {
    rows: usize,
    n_in: usize,
    n_out: usize,
    algo: LtAlgo,
}

/// A [`Pick`] from the probe's printed form: `[id, tile, stages, splitk, reduction,
/// swizzle, custom, inner, cluster]`
const fn pick(rows: usize, n_in: usize, n_out: usize, c: [u32; 9]) -> Pick {
    Pick {
        rows,
        n_in,
        n_out,
        algo: LtAlgo {
            id: c[0] as i32,
            tile: c[1],
            stages: c[2],
            splitk: c[3],
            reduction: c[4],
            swizzle: c[5],
            custom: c[6],
            inner: c[7] as u16,
            cluster: c[8] as u16,
        },
    }
}

/// Measured picks for the decode buckets, from gpu0 job 5131eee3 (RTX 5090, cuBLAS
/// 12.8.4, weights rotated past L2): each is the fastest deterministic candidate whose
/// error against an exact f64 product stays within 1.25x torch's max and 1.05x its mean,
/// kept where it beat torch's choice by 3% or more, with the per-launch times in the
/// comments. Below 32 rows every split-K candidate that would beat torch adds about a
/// third to the mean error, as if the product were rounded to FP16 before the bias, so
/// those shapes keep torch's choice. Shapes not listed run torch's choice. The table is
/// tied to this cuBLAS release and was measured only on the RTX 5090; another release or
/// GPU should be re-probed
const TABLE: &[Pick] = &[
    pick(32, 1280, 3840, [21, 11, 20, 1, 0, 0, 0, 0, 0]), // 35.49 -> 9.02 us
    pick(42, 1280, 3840, [21, 11, 20, 1, 0, 0, 0, 0, 0]), // 36.72 -> 9.31 us
    pick(52, 1280, 3840, [21, 11, 20, 1, 0, 0, 0, 0, 0]), // 36.88 -> 9.55 us
    pick(54, 1280, 3840, [21, 11, 20, 1, 0, 0, 0, 0, 0]), // 36.90 -> 9.60 us
    pick(32, 1280, 1280, [21, 11, 13, 20, 2, 0, 0, 0, 0]), // 6.45 -> 5.86 us
    pick(42, 1280, 1280, [21, 11, 20, 4, 2, 0, 0, 0, 0]), // 6.51 -> 5.98 us
    pick(52, 1280, 1280, [21, 15, 12, 4, 2, 0, 0, 0, 0]), // 6.58 -> 6.02 us
    pick(54, 1280, 1280, [21, 15, 12, 4, 2, 0, 0, 0, 0]), // 6.61 -> 6.02 us
    pick(1, 1280, 5120, [21, 11, 20, 1, 0, 0, 0, 0, 0]),  // 10.98 -> 10.32 us
    pick(32, 1280, 5120, [21, 11, 20, 1, 0, 0, 0, 0, 0]), // 18.43 -> 10.48 us
    pick(42, 1280, 5120, [21, 11, 20, 1, 0, 0, 0, 0, 0]), // 18.46 -> 10.83 us
    pick(52, 1280, 5120, [21, 15, 12, 1, 0, 0, 0, 0, 0]), // 18.46 -> 11.04 us
    pick(54, 1280, 5120, [21, 15, 12, 1, 0, 0, 0, 0, 0]), // 18.46 -> 11.02 us
    pick(1, 5120, 1280, [13, 0, 0, 1, 0, 0, 75, 0, 0]),   // 11.78 -> 10.54 us
    pick(32, 5120, 1280, [21, 11, 14, 12, 2, 0, 0, 0, 0]), // 28.77 -> 11.42 us
    pick(42, 5120, 1280, [21, 11, 14, 8, 2, 0, 0, 0, 0]), // 22.56 -> 11.79 us
    pick(52, 5120, 1280, [21, 15, 12, 5, 2, 0, 0, 0, 0]), // 35.81 -> 11.90 us
    pick(54, 5120, 1280, [21, 15, 12, 5, 2, 0, 0, 0, 0]), // 36.00 -> 11.90 us
];

/// Split counts the candidate sweep tries on each heuristic algorithm
const SPLITS: [u32; 12] = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24];

type Attr = lt::cublasLtMatmulAlgoConfigAttributes_t;

impl LtAlgo {
    fn read(f: &LtFns, algo: &lt::cublasLtMatmulAlgo_t) -> Result<Self> {
        fn get<T: Default>(f: &LtFns, algo: &lt::cublasLtMatmulAlgo_t, attr: Attr) -> Result<T> {
            let mut v = T::default();
            let mut written = 0usize;
            check_lt(
                unsafe {
                    (f.algo_config_get)(
                        algo,
                        attr,
                        (&mut v as *mut T).cast(),
                        size_of::<T>(),
                        &mut written,
                    )
                },
                "algo config attribute",
            )?;
            Ok(v)
        }

        Ok(Self {
            id: get(f, algo, Attr::CUBLASLT_ALGO_CONFIG_ID)?,
            tile: get(f, algo, Attr::CUBLASLT_ALGO_CONFIG_TILE_ID)?,
            stages: get(f, algo, Attr::CUBLASLT_ALGO_CONFIG_STAGES_ID)?,
            splitk: get(f, algo, Attr::CUBLASLT_ALGO_CONFIG_SPLITK_NUM)?,
            reduction: get(f, algo, Attr::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)?,
            swizzle: get(f, algo, Attr::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING)?,
            custom: get(f, algo, Attr::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION)?,
            inner: get(f, algo, Attr::CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID)?,
            cluster: get(f, algo, Attr::CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID)?,
        })
    }

    /// The library's algorithm value for this configuration (FP16 operands, FP32 compute
    /// and scale), not yet checked against a shape
    fn build(&self, blas: &Blas) -> Result<lt::cublasLtMatmulAlgo_t> {
        let f = &blas.libs.lt;
        let f16 = lt::cudaDataType_t::CUDA_R_16F;
        let mut algo: lt::cublasLtMatmulAlgo_t = unsafe { std::mem::zeroed() };
        check_lt(
            unsafe {
                (f.algo_init)(
                    blas.lt,
                    lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    lt::cudaDataType_t::CUDA_R_32F,
                    f16,
                    f16,
                    f16,
                    f16,
                    self.id,
                    &mut algo,
                )
            },
            "cublasLtMatmulAlgoInit",
        )?;
        fn set<T>(f: &LtFns, algo: &mut lt::cublasLtMatmulAlgo_t, attr: Attr, v: T) -> Result<()> {
            check_lt(
                unsafe { (f.algo_config_set)(algo, attr, (&v as *const T).cast(), size_of::<T>()) },
                "algo config attribute",
            )
        }

        set(f, &mut algo, Attr::CUBLASLT_ALGO_CONFIG_TILE_ID, self.tile)?;
        set(
            f,
            &mut algo,
            Attr::CUBLASLT_ALGO_CONFIG_STAGES_ID,
            self.stages,
        )?;
        set(
            f,
            &mut algo,
            Attr::CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
            self.splitk,
        )?;
        set(
            f,
            &mut algo,
            Attr::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
            self.reduction,
        )?;
        set(
            f,
            &mut algo,
            Attr::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
            self.swizzle,
        )?;
        set(
            f,
            &mut algo,
            Attr::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
            self.custom,
        )?;
        set(
            f,
            &mut algo,
            Attr::CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
            self.inner,
        )?;
        set(
            f,
            &mut algo,
            Attr::CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
            self.cluster,
        )?;
        Ok(algo)
    }

    /// Whether the split-K partials, if any, are summed in FP32 by a separate pass: the
    /// in-place and output-type schemes add them up in FP16
    fn fp32_reduction(&self) -> bool {
        self.splitk <= 1
            || self.reduction
                == lt::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE as u32
    }
}

/// Whether `gpu` is the device [`TABLE`] was measured on, an RTX 5090 (compute 12.0,
/// 170 SMs): split counts and tiles were picked for its SM count, so other GPUs keep
/// torch's choice until they are probed
pub(super) fn measured_on(gpu: &Gpu) -> bool {
    gpu.compute_capability() == (12, 0) && gpu.sms() == 170
}

fn key(x: u64, w: u64, y: u64, (rows, n_in, n_out): (usize, usize, usize)) -> LinearKey {
    LinearKey {
        rows,
        n_in,
        n_out,
        align: [alignment(w), alignment(x), alignment(y), alignment(y)],
    }
}

impl Blas {
    /// [`Blas::linear_bias`] for a decode step: `y[rows, n_out] = x[rows, n_in] @ w[n_out,
    /// n_in]^T + bias` on the measured algorithm for the shape when [`TABLE`] has one, the
    /// GPU is the one it was measured on and every operand is 16-byte aligned, else
    /// exactly as torch runs it
    pub fn linear_bias_decode(
        &self,
        x: u64,
        w: u64,
        bias: u64,
        y: u64,
        dims: (usize, usize, usize),
    ) -> Result<()> {
        let (rows, n_in, n_out) = dims;
        let aligned = [x, w, bias, y].iter().all(|p| p.is_multiple_of(16));
        let pick = TABLE
            .iter()
            .filter(|_| self.decode_table)
            .find(|p| p.rows == rows && p.n_in == n_in && p.n_out == n_out);
        match pick {
            Some(p) if aligned => self.linear_bias_with(x, w, bias, y, dims, p.algo),
            _ => self.linear_bias(x, w, bias, y, dims),
        }
    }

    /// The algorithms cuBLASLt accepts for a linear with bias of this shape and these
    /// operand alignments: up to `max` heuristic results under the decode workspace,
    /// plus each one's split-K variants with the FP32 reduction, deduplicated. For the
    /// probe that measures [`TABLE`]
    pub fn linear_bias_candidates(
        &self,
        (x, w, y): (u64, u64, u64),
        dims: (usize, usize, usize),
        max: usize,
    ) -> Result<Vec<LtCandidate>> {
        let key = key(x, w, y, dims);
        let plan = self.plan_linear(key)?;
        let out = self.candidates_on(&plan, key, max);
        self.destroy_plan(plan);
        out
    }

    fn candidates_on(
        &self,
        plan: &LinearPlan,
        key: LinearKey,
        max: usize,
    ) -> Result<Vec<LtCandidate>> {
        let found = self.heuristic_all(plan, key, max)?;
        let mut out: Vec<LtCandidate> = Vec::new();
        let mut push = |c: LtCandidate| {
            if c.algo.fp32_reduction() && !out.iter().any(|o| o.algo == c.algo) {
                out.push(c);
            }
        };
        for &(algo, workspace, waves) in &found {
            push(LtCandidate {
                algo,
                workspace,
                waves,
                heuristic: true,
            });
        }

        for (base, _, _) in &found {
            for splitk in SPLITS {
                let reduction = if splitk == 1 {
                    lt::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_NONE
                } else {
                    lt::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE
                } as u32;
                let algo = LtAlgo {
                    splitk,
                    reduction,
                    ..*base
                };
                if let Ok(workspace) = self.check(plan, &algo)
                    && workspace <= DECODE_WORKSPACE
                {
                    push(LtCandidate {
                        algo,
                        workspace,
                        waves: 0.0,
                        heuristic: false,
                    });
                }
            }
        }

        Ok(out)
    }

    /// Up to `max` heuristic results for the plan's shape under the decode workspace
    fn heuristic_all(
        &self,
        plan: &LinearPlan,
        key: LinearKey,
        max: usize,
    ) -> Result<Vec<(LtAlgo, usize, f32)>> {
        let f = &self.libs.lt;
        unsafe {
            let mut pref = std::ptr::null_mut();
            check_lt((f.pref_create)(&mut pref), "preference")?;
            let ws = DECODE_WORKSPACE as u64;
            let mut status = (f.pref_set)(
                pref,
                lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&ws as *const u64).cast(),
                size_of::<u64>(),
            );
            let attrs = [
                lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
            ];
            for (attr, a) in attrs.into_iter().zip(key.align) {
                if status == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    status = (f.pref_set)(pref, attr, (&a as *const u32).cast(), size_of::<u32>());
                }
            }

            let mut results: Vec<lt::cublasLtMatmulHeuristicResult_t> =
                vec![std::mem::zeroed(); max];
            let mut found = 0;
            if status == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                status = (f.heuristic)(
                    self.lt,
                    plan.desc,
                    plan.w,
                    plan.x,
                    plan.y,
                    plan.y,
                    pref,
                    i32::try_from(max)?,
                    results.as_mut_ptr(),
                    &mut found,
                );
            }
            (f.pref_destroy)(pref);
            check_lt(status, "cublasLtMatmulAlgoGetHeuristic")?;

            results
                .iter()
                .take(found.max(0) as usize)
                .filter(|r| r.state == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS)
                .map(|r| Ok((LtAlgo::read(f, &r.algo)?, r.workspaceSize, r.wavesCount)))
                .collect()
        }
    }

    /// The workspace `algo` needs on the plan's shape, or an error if the library
    /// doesn't support it there
    fn check(&self, plan: &LinearPlan, algo: &LtAlgo) -> Result<usize> {
        let built = algo.build(self)?;
        let f = &self.libs.lt;
        let mut result: lt::cublasLtMatmulHeuristicResult_t = unsafe { std::mem::zeroed() };
        check_lt(
            unsafe {
                (f.algo_check)(
                    self.lt,
                    plan.desc,
                    plan.w,
                    plan.x,
                    plan.y,
                    plan.y,
                    &built,
                    &mut result,
                )
            },
            "cublasLtMatmulAlgoCheck",
        )?;
        check_lt(result.state, "cublasLtMatmulAlgoCheck state")?;
        Ok(result.workspaceSize)
    }

    fn destroy_plan(&self, plan: LinearPlan) {
        let f = &self.libs.lt;
        unsafe {
            (f.layout_destroy)(plan.w);
            (f.layout_destroy)(plan.x);
            (f.layout_destroy)(plan.y);
            (f.desc_destroy)(plan.desc);
        }
    }

    /// [`Blas::linear_bias`] on the given algorithm with the decode workspace. Fails if
    /// the library rejects the algorithm for this shape or it needs more workspace
    pub fn linear_bias_with(
        &self,
        x: u64,
        w: u64,
        bias: u64,
        y: u64,
        dims: (usize, usize, usize),
        algo: LtAlgo,
    ) -> Result<()> {
        ensure!(
            algo.fp32_reduction(),
            "algorithm {algo:?} sums split-K partials in FP16"
        );
        let key = key(x, w, y, dims);
        let mut plans = self.algo_plans.borrow_mut();
        let plan = match plans.entry((key, algo)) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => {
                let mut plan = self.plan_linear(key)?;
                let checked = self.check(&plan, &algo).and_then(|ws| {
                    if ws > DECODE_WORKSPACE {
                        bail!("{algo:?} needs {ws} bytes of workspace, over {DECODE_WORKSPACE}")
                    }
                    algo.build(self)
                });
                match checked {
                    Ok(built) => plan.algo = built,
                    Err(err) => {
                        self.destroy_plan(plan);
                        return Err(err.wrap_err(format!("linear {dims:?} on {algo:?}")));
                    }
                }
                e.insert(plan)
            }
        };

        let f = &self.libs.lt;
        unsafe {
            check_lt(
                (f.desc_set)(
                    plan.desc,
                    lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                    (&bias as *const u64).cast(),
                    size_of::<u64>(),
                ),
                "set bias pointer",
            )?;
            let (alpha, beta) = (1f32, 0f32);
            check_lt(
                (f.matmul)(
                    self.lt,
                    plan.desc,
                    (&alpha as *const f32).cast(),
                    w as *const c_void,
                    plan.w,
                    x as *const c_void,
                    plan.x,
                    (&beta as *const f32).cast(),
                    y as *const c_void,
                    plan.y,
                    y as *mut c_void,
                    plan.y,
                    &plan.algo,
                    self.decode_workspace.ptr() as *mut c_void,
                    DECODE_WORKSPACE,
                    self.stream as _,
                ),
                "cublasLtMatmul",
            )?;
        }

        Ok(())
    }
}
