//! The absorbed cross-attention's GEMMs: the per-head query and value projections as
//! strided-batched cuBLASLt matmuls
//!
//! None of these has a torch counterpart (the reference projects keys and values instead),
//! so they take cuBLASLt's heuristic choice for their own shapes

use std::collections::hash_map::Entry;
use std::ffi::c_void;

use cudarc::cublas::sys as blas;
use cudarc::cublaslt::sys as lt;
use eyre::{Result, ensure};

use super::{Blas, LT_WORKSPACE, check_lt};

/// One operand of a strided-batched matmul, column-major as cuBLAS sees it: `rows x cols`
/// with leading dimension `ld`, entry `i` at `ptr + 2 * i * stride` bytes, all FP16
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Strided {
    pub rows: usize,
    pub cols: usize,
    pub ld: usize,
    pub stride: usize,
}

/// Shape and epilogue of a strided-batched FP16 matmul `d_i = alpha op(a_i) b_i (+ bias_i)`
/// with FP32 compute
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BatchedKey {
    pub a: Strided,
    /// Whether `a` is transposed (`op(a) = a^T`)
    pub trans_a: bool,
    pub b: Strided,
    pub d: Strided,
    pub batch: usize,
    /// Elements between consecutive entries' bias vectors, when there is a bias epilogue
    pub bias_stride: Option<usize>,
}

/// Descriptors and the heuristic's algorithm for one [`BatchedKey`]
pub(super) struct BatchedPlan {
    desc: lt::cublasLtMatmulDesc_t,
    layouts: [lt::cublasLtMatrixLayout_t; 3],
    algo: lt::cublasLtMatmulAlgo_t,
}

impl BatchedPlan {
    pub(super) fn destroy(&self, blas: &Blas) {
        let f = &blas.libs.lt;
        unsafe {
            for l in self.layouts {
                (f.layout_destroy)(l);
            }
            (f.desc_destroy)(self.desc);
        }
    }
}

impl Blas {
    /// `d_i = alpha op(a_i) @ b_i (+ bias_i)` for `key.batch` entries, FP16 operands and
    /// FP32 compute, the bias added to each row of `d_i` in the FP32 epilogue; `bias` is
    /// read only when the key has a bias stride
    pub fn matmul_batched(
        &self,
        key: BatchedKey,
        alpha: f32,
        (a, b, d): (u64, u64, u64),
        bias: u64,
    ) -> Result<()> {
        let (m, k) = if key.trans_a {
            (key.a.cols, key.a.rows)
        } else {
            (key.a.rows, key.a.cols)
        };
        ensure!(
            key.b.rows == k && key.d.rows == m && key.d.cols == key.b.cols && key.batch > 0,
            "batched matmul shapes don't chain: {key:?}"
        );
        let mut plans = self.batched_plans.borrow_mut();
        let plan = match plans.entry(key) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(self.plan_batched(key)?),
        };
        let f = &self.libs.lt;
        unsafe {
            if key.bias_stride.is_some() {
                check_lt(
                    (f.desc_set)(
                        plan.desc,
                        lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                        (&bias as *const u64).cast(),
                        size_of::<u64>(),
                    ),
                    "set bias pointer",
                )?;
            }
            let beta = 0f32;
            let [la, lb, ld] = plan.layouts;
            check_lt(
                (f.matmul)(
                    self.lt,
                    plan.desc,
                    (&alpha as *const f32).cast(),
                    a as *const c_void,
                    la,
                    b as *const c_void,
                    lb,
                    (&beta as *const f32).cast(),
                    d as *const c_void,
                    ld,
                    d as *mut c_void,
                    ld,
                    &plan.algo,
                    self.lt_workspace.ptr() as *mut c_void,
                    LT_WORKSPACE,
                    self.stream as _,
                ),
                "cublasLtMatmul (batched)",
            )
        }
    }

    fn plan_batched(&self, key: BatchedKey) -> Result<BatchedPlan> {
        let f = &self.libs.lt;
        unsafe {
            let mut desc = std::ptr::null_mut();
            check_lt(
                (f.desc_create)(
                    &mut desc,
                    lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    lt::cudaDataType_t::CUDA_R_32F,
                ),
                "cublasLtMatmulDescCreate",
            )?;
            let set_i32 = |attr, value: i32| {
                check_lt(
                    (f.desc_set)(desc, attr, (&value as *const i32).cast(), size_of::<i32>()),
                    "desc attribute",
                )
            };
            let op = |t: bool| {
                if t {
                    blas::cublasOperation_t::CUBLAS_OP_T as i32
                } else {
                    blas::cublasOperation_t::CUBLAS_OP_N as i32
                }
            };
            set_i32(
                lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                op(key.trans_a),
            )?;
            set_i32(
                lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                op(false),
            )?;
            if let Some(stride) = key.bias_stride {
                set_i32(
                    lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
                    lt::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS as i32,
                )?;
                let s = i64::try_from(stride)?;
                check_lt(
                    (f.desc_set)(
                        desc,
                        lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE,
                        (&s as *const i64).cast(),
                        size_of::<i64>(),
                    ),
                    "bias batch stride",
                )?;
            }

            let batch = i32::try_from(key.batch)?;
            let layout = |o: Strided| -> Result<lt::cublasLtMatrixLayout_t> {
                let mut l = std::ptr::null_mut();
                check_lt(
                    (f.layout_create)(
                        &mut l,
                        lt::cudaDataType_t::CUDA_R_16F,
                        o.rows as u64,
                        o.cols as u64,
                        i64::try_from(o.ld)?,
                    ),
                    "layout",
                )?;
                let stride = i64::try_from(o.stride)?;
                for (attr, ptr, size) in [
                    (
                        lt::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                        (&batch as *const i32).cast::<c_void>(),
                        size_of::<i32>(),
                    ),
                    (
                        lt::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                        (&stride as *const i64).cast::<c_void>(),
                        size_of::<i64>(),
                    ),
                ] {
                    check_lt(
                        (f.layout_set)(l, attr, ptr, size),
                        "batched layout attribute",
                    )?;
                }
                Ok(l)
            };
            let layouts = [layout(key.a)?, layout(key.b)?, layout(key.d)?];

            let mut pref = std::ptr::null_mut();
            check_lt((f.pref_create)(&mut pref), "preference")?;
            let ws = LT_WORKSPACE as u64;
            check_lt(
                (f.pref_set)(
                    pref,
                    lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    (&ws as *const u64).cast(),
                    size_of::<u64>(),
                ),
                "workspace preference",
            )?;
            let mut result: lt::cublasLtMatmulHeuristicResult_t = std::mem::zeroed();
            let mut found = 0;
            let [la, lb, ld] = layouts;
            let status = (f.heuristic)(
                self.lt,
                desc,
                la,
                lb,
                ld,
                ld,
                pref,
                1,
                &mut result,
                &mut found,
            );
            (f.pref_destroy)(pref);
            check_lt(status, "cublasLtMatmulAlgoGetHeuristic (batched)")?;
            ensure!(
                found > 0,
                "no cuBLASLt algorithm for batched matmul {key:?}"
            );
            Ok(BatchedPlan {
                desc,
                layouts,
                algo: result.algo,
            })
        }
    }
}
