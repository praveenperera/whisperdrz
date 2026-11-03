//! cuBLAS and cuBLASLt called the way torch 2.8 calls them, so every GEMM picks the same
//! kernel and gives the same bits as the Python reference
//!
//! The libraries are opened from an explicit directory (`WHISPERDRZ_CUBLAS_DIR`, else the
//! dynamic loader's `libcublasLt.so.12`) and must be exactly cuBLAS 12.8.4, the build in
//! torch's wheel: the heuristic's kernel choice, and so the summation order, can change in
//! any other release. cudarc only provides the types; its own loader would look for an
//! unversioned `libcublas.so` first and could open a toolkit's cuBLAS 13 instead.
//! `rust/docs/torch-gemm-calls.md` lists the calls torch makes

use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::ffi::{c_int, c_void};
use std::path::PathBuf;

use cudarc::cublas::sys as blas;
use cudarc::cublaslt::sys as lt;
use eyre::{Result, WrapErr, bail, ensure};
use libloading::Library;

use cudarc::driver::sys::CUstream;

use super::device::{Buf, Gpu};

/// The cuBLAS release torch 2.8.0+cu128 ships, as `cublasLtGetVersion` reports it
const CUBLAS_VERSION: usize = 120804;

/// torch's cuBLASLt workspace for `addmm` (`workSpaceSizeInBytes` in its calls)
const LT_WORKSPACE: usize = 1 << 20;

/// torch's handle workspace for `cublasGemmEx` (`cublasSetWorkspace_v2` in its calls)
const BLAS_WORKSPACE: usize = 8519680;

/// Declares a table of C entry points and the code that resolves them from a library
macro_rules! entry_points {
    ($table:ident -> $status:ty { $($field:ident = $symbol:literal: fn($($arg:ty),* $(,)?) $(-> $ret:ty)?;)* }) => {
        struct $table {
            $($field: unsafe extern "C" fn($($arg),*) -> entry_points!(@ret $status $(, $ret)?),)*
        }

        impl $table {
            /// # Safety
            /// `lib` must be the library the table is declared for, so each symbol has
            /// the signature given here
            unsafe fn resolve(lib: &Library) -> Result<Self> {
                Ok(Self {
                    $($field: *unsafe { lib.get(concat!($symbol, "\0").as_bytes()) }
                        .wrap_err(concat!("missing ", $symbol))?,)*
                })
            }
        }
    };
    (@ret $status:ty) => { $status };
    (@ret $status:ty, $ret:ty) => { $ret };
}

entry_points!(BlasFns -> blas::cublasStatus_t {
    create = "cublasCreate_v2": fn(*mut blas::cublasHandle_t);
    destroy = "cublasDestroy_v2": fn(blas::cublasHandle_t);
    set_stream = "cublasSetStream_v2": fn(blas::cublasHandle_t, blas::cudaStream_t);
    set_math_mode = "cublasSetMathMode": fn(blas::cublasHandle_t, blas::cublasMath_t);
    set_workspace = "cublasSetWorkspace_v2": fn(blas::cublasHandle_t, *mut c_void, usize);
    gemm_ex = "cublasGemmEx": fn(
        blas::cublasHandle_t,
        blas::cublasOperation_t,
        blas::cublasOperation_t,
        c_int,
        c_int,
        c_int,
        *const c_void,
        *const c_void,
        blas::cudaDataType,
        c_int,
        *const c_void,
        blas::cudaDataType,
        c_int,
        *const c_void,
        *mut c_void,
        blas::cudaDataType,
        c_int,
        blas::cublasComputeType_t,
        blas::cublasGemmAlgo_t,
    );
});

entry_points!(LtFns -> lt::cublasStatus_t {
    version = "cublasLtGetVersion": fn() -> usize;
    create = "cublasLtCreate": fn(*mut lt::cublasLtHandle_t);
    destroy = "cublasLtDestroy": fn(lt::cublasLtHandle_t);
    desc_create = "cublasLtMatmulDescCreate":
        fn(*mut lt::cublasLtMatmulDesc_t, lt::cublasComputeType_t, lt::cudaDataType_t);
    desc_destroy = "cublasLtMatmulDescDestroy": fn(lt::cublasLtMatmulDesc_t);
    desc_set = "cublasLtMatmulDescSetAttribute":
        fn(lt::cublasLtMatmulDesc_t, lt::cublasLtMatmulDescAttributes_t, *const c_void, usize);
    layout_create = "cublasLtMatrixLayoutCreate":
        fn(*mut lt::cublasLtMatrixLayout_t, lt::cudaDataType, u64, u64, i64);
    layout_destroy = "cublasLtMatrixLayoutDestroy": fn(lt::cublasLtMatrixLayout_t);
    pref_create = "cublasLtMatmulPreferenceCreate": fn(*mut lt::cublasLtMatmulPreference_t);
    pref_destroy = "cublasLtMatmulPreferenceDestroy": fn(lt::cublasLtMatmulPreference_t);
    pref_set = "cublasLtMatmulPreferenceSetAttribute":
        fn(lt::cublasLtMatmulPreference_t, lt::cublasLtMatmulPreferenceAttributes_t, *const c_void, usize);
    heuristic = "cublasLtMatmulAlgoGetHeuristic": fn(
        lt::cublasLtHandle_t,
        lt::cublasLtMatmulDesc_t,
        lt::cublasLtMatrixLayout_t,
        lt::cublasLtMatrixLayout_t,
        lt::cublasLtMatrixLayout_t,
        lt::cublasLtMatrixLayout_t,
        lt::cublasLtMatmulPreference_t,
        c_int,
        *mut lt::cublasLtMatmulHeuristicResult_t,
        *mut c_int,
    );
    matmul = "cublasLtMatmul": fn(
        lt::cublasLtHandle_t,
        lt::cublasLtMatmulDesc_t,
        *const c_void,
        *const c_void,
        lt::cublasLtMatrixLayout_t,
        *const c_void,
        lt::cublasLtMatrixLayout_t,
        *const c_void,
        *const c_void,
        lt::cublasLtMatrixLayout_t,
        *mut c_void,
        lt::cublasLtMatrixLayout_t,
        *const lt::cublasLtMatmulAlgo_t,
        *mut c_void,
        usize,
        lt::cudaStream_t,
    );
});

/// cuBLAS and cuBLASLt from one install, open for the life of the value
struct Libs {
    blas: BlasFns,
    lt: LtFns,
    // the entry points point into these
    _libs: [Library; 2],
}

impl Libs {
    fn open() -> Result<Self> {
        let dir = std::env::var_os("WHISPERDRZ_CUBLAS_DIR").map(PathBuf::from);
        let path = |soname: &str| {
            dir.as_ref()
                .map_or_else(|| PathBuf::from(soname), |d| d.join(soname))
        };
        let open = |soname: &str| {
            let p = path(soname);
            unsafe { Library::new(&p) }
                .wrap_err_with(|| format!("open {} (set WHISPERDRZ_CUBLAS_DIR)", p.display()))
        };

        // libcublas binds libcublasLt by soname, so opening Lt first makes it use this copy
        let lt_lib = open("libcublasLt.so.12")?;
        let blas_lib = open("libcublas.so.12")?;
        let lt = unsafe { LtFns::resolve(&lt_lib) }?;
        let blas = unsafe { BlasFns::resolve(&blas_lib) }?;
        let version = unsafe { (lt.version)() };
        ensure!(
            version == CUBLAS_VERSION,
            "cuBLASLt {version} loaded; parity with torch 2.8.0+cu128 needs exactly {CUBLAS_VERSION} \
             (pip nvidia-cublas-cu12==12.8.4.1)"
        );
        Ok(Self {
            blas,
            lt,
            _libs: [blas_lib, lt_lib],
        })
    }
}

fn check_blas(status: blas::cublasStatus_t, what: &str) -> Result<()> {
    if status != blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        bail!("{what}: cuBLAS error {status:?}");
    }

    Ok(())
}

fn check_lt(status: lt::cublasStatus_t, what: &str) -> Result<()> {
    if status != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        bail!("{what}: cuBLASLt error {status:?}");
    }

    Ok(())
}

/// torch's `_getAlignment`: the largest power of two up to 256 dividing the address
fn alignment(ptr: u64) -> u32 {
    let mut a = 256u32;
    while a > 1 && !ptr.is_multiple_of(a as u64) {
        a /= 2;
    }

    a
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct LinearKey {
    rows: usize,
    n_in: usize,
    n_out: usize,
    align: [u32; 4],
}

/// Descriptors and the heuristic's algorithm for one linear shape
struct LinearPlan {
    desc: lt::cublasLtMatmulDesc_t,
    w: lt::cublasLtMatrixLayout_t,
    x: lt::cublasLtMatrixLayout_t,
    y: lt::cublasLtMatrixLayout_t,
    algo: lt::cublasLtMatmulAlgo_t,
}

/// GEMMs on the GPU's stream with torch's descriptors
pub struct Blas {
    libs: Libs,
    lt: lt::cublasLtHandle_t,
    handle: blas::cublasHandle_t,
    stream: CUstream,
    lt_workspace: Buf,
    _blas_workspace: Buf,
    plans: RefCell<HashMap<LinearKey, LinearPlan>>,
}

impl Blas {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let libs = Libs::open()?;
        let lt_workspace = gpu.alloc(LT_WORKSPACE)?;
        let blas_workspace = gpu.alloc(BLAS_WORKSPACE)?;

        let mut lt = std::ptr::null_mut();
        check_lt(unsafe { (libs.lt.create)(&mut lt) }, "cublasLtCreate")?;
        let mut handle = std::ptr::null_mut();
        check_blas(unsafe { (libs.blas.create)(&mut handle) }, "cublasCreate")?;
        let this = Self {
            libs,
            lt,
            handle,
            stream: gpu.cu_stream(),
            lt_workspace,
            _blas_workspace: blas_workspace,
            plans: RefCell::default(),
        };

        let fns = &this.libs.blas;
        unsafe {
            check_blas(
                (fns.set_stream)(handle, gpu.cu_stream() as _),
                "cublasSetStream",
            )?;
            // torch runs with allow_fp16_reduced_precision_reduction=True, which is default math
            check_blas(
                (fns.set_math_mode)(handle, blas::cublasMath_t::CUBLAS_DEFAULT_MATH),
                "cublasSetMathMode",
            )?;
            check_blas(
                (fns.set_workspace)(
                    handle,
                    this._blas_workspace.ptr() as *mut c_void,
                    BLAS_WORKSPACE,
                ),
                "cublasSetWorkspace",
            )?;
        }

        Ok(this)
    }

    /// `y[rows, n_out] = x[rows, n_in] @ w[n_out, n_in]^T + bias`, all FP16 row-major, as
    /// torch's `addmm` runs it: cuBLASLt with a bias epilogue and FP32 compute
    pub fn linear_bias(
        &self,
        x: u64,
        w: u64,
        bias: u64,
        y: u64,
        (rows, n_in, n_out): (usize, usize, usize),
    ) -> Result<()> {
        let key = LinearKey {
            rows,
            n_in,
            n_out,
            align: [alignment(w), alignment(x), alignment(y), alignment(y)],
        };
        let mut plans = self.plans.borrow_mut();
        let plan = match plans.entry(key) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(self.plan_linear(key)?),
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
                    self.lt_workspace.ptr() as *mut c_void,
                    LT_WORKSPACE,
                    self.stream as _,
                ),
                "cublasLtMatmul",
            )?;
        }

        Ok(())
    }

    fn plan_linear(&self, key: LinearKey) -> Result<LinearPlan> {
        let f = &self.libs.lt;
        let f16 = lt::cudaDataType_t::CUDA_R_16F;
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
            let set = |attr, value: i32| {
                check_lt(
                    (f.desc_set)(desc, attr, (&value as *const i32).cast(), size_of::<i32>()),
                    "desc attribute",
                )
            };
            set(
                lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                blas::cublasOperation_t::CUBLAS_OP_T as i32,
            )?;
            set(
                lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                blas::cublasOperation_t::CUBLAS_OP_N as i32,
            )?;
            set(
                lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
                lt::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS as i32,
            )?;

            // column-major views of the row-major operands, as torch builds them
            let layout =
                |rows: usize, cols: usize, ld: usize| -> Result<lt::cublasLtMatrixLayout_t> {
                    let mut l = std::ptr::null_mut();
                    check_lt(
                        (f.layout_create)(&mut l, f16, rows as u64, cols as u64, ld as i64),
                        "layout",
                    )?;
                    Ok(l)
                };
            let w = layout(key.n_in, key.n_out, key.n_in)?;
            let x = layout(key.n_in, key.rows, key.n_in)?;
            let y = layout(key.n_out, key.rows, key.n_out)?;

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
            let attrs = [
                lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
            ];
            for (attr, a) in attrs.into_iter().zip(key.align) {
                check_lt(
                    (f.pref_set)(pref, attr, (&a as *const u32).cast(), size_of::<u32>()),
                    "alignment preference",
                )?;
            }

            let mut result: lt::cublasLtMatmulHeuristicResult_t = std::mem::zeroed();
            let mut found = 0;
            let status = (f.heuristic)(self.lt, desc, w, x, y, y, pref, 1, &mut result, &mut found);
            (f.pref_destroy)(pref);
            check_lt(status, "cublasLtMatmulAlgoGetHeuristic")?;
            ensure!(
                found > 0,
                "no cuBLASLt algorithm for linear {}x{} -> {}",
                key.rows,
                key.n_in,
                key.n_out
            );
            Ok(LinearPlan {
                desc,
                w,
                x,
                y,
                algo: result.algo,
            })
        }
    }

    /// `cublasGemmEx` with FP16 operands and FP32 compute on the handle torch configures:
    /// column-major `c[m, n] = op(a) @ op(b)`
    pub fn gemm_ex_f16(
        &self,
        trans_a: bool,
        trans_b: bool,
        (m, n, k): (usize, usize, usize),
        a: (u64, usize),
        b: (u64, usize),
        c: (u64, usize),
    ) -> Result<()> {
        let op = |t: bool| {
            if t {
                blas::cublasOperation_t::CUBLAS_OP_T
            } else {
                blas::cublasOperation_t::CUBLAS_OP_N
            }
        };
        let f16 = blas::cudaDataType::CUDA_R_16F;
        let (alpha, beta) = (1f32, 0f32);
        let dim = |v: usize| c_int::try_from(v).wrap_err("GEMM dimension overflows int");
        unsafe {
            check_blas(
                (self.libs.blas.gemm_ex)(
                    self.handle,
                    op(trans_a),
                    op(trans_b),
                    dim(m)?,
                    dim(n)?,
                    dim(k)?,
                    (&alpha as *const f32).cast(),
                    a.0 as *const c_void,
                    f16,
                    dim(a.1)?,
                    b.0 as *const c_void,
                    f16,
                    dim(b.1)?,
                    (&beta as *const f32).cast(),
                    c.0 as *mut c_void,
                    f16,
                    dim(c.1)?,
                    blas::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    blas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                ),
                "cublasGemmEx",
            )
        }
    }

    /// `y[rows, n_out] = x[rows, n_in] @ w[n_out, n_in]^T` without bias, as torch's `mm`
    /// runs a bias-free `nn.Linear`
    pub fn linear(
        &self,
        x: u64,
        w: u64,
        y: u64,
        (rows, n_in, n_out): (usize, usize, usize),
    ) -> Result<()> {
        self.gemm_ex_f16(
            true,
            false,
            (n_out, rows, n_in),
            (w, n_in),
            (x, n_in),
            (y, n_out),
        )
    }

    /// `y[rows, n] = x[rows, k] @ b[k, n]`, all FP16 row-major, as torch's `mm` runs the
    /// LM head on the contiguous transposed token embedding
    pub fn matmul(
        &self,
        x: u64,
        b: u64,
        y: u64,
        (rows, k, n): (usize, usize, usize),
    ) -> Result<()> {
        self.gemm_ex_f16(false, false, (n, rows, k), (b, n), (x, k), (y, n))
    }
}

impl Drop for Blas {
    fn drop(&mut self) {
        let f = &self.libs.lt;
        unsafe {
            for plan in self.plans.borrow().values() {
                (f.layout_destroy)(plan.w);
                (f.layout_destroy)(plan.x);
                (f.layout_destroy)(plan.y);
                (f.desc_destroy)(plan.desc);
            }

            (f.destroy)(self.lt);
            (self.libs.blas.destroy)(self.handle);
        }
    }
}
