# The GEMM calls torch makes, and how the native runtime repeats them

Captured with `CUBLASLT_LOG_LEVEL=5`, `CUBLAS_LOGINFO_DBG=1` and `CUDNN_LOGLEVEL_DBG=3`
on one 60 s VoxConverse clip (willh) through the padding-fixed pipeline, torch 2.8.0+cu128,
cuBLAS 12.8.4, cuDNN 9.10.2, RTX 5090 (gpu0 job a8982308). cuBLAS picks a kernel per shape
from its heuristic, and the kernel fixes the summation order, so calling the same library
build with the same descriptors, preferences and workspace sizes gives the same bits.

## Linears with bias: cuBLASLt matmul

`nn.Linear` with bias on FP16 runs `cublasLtMatmul` for every row count, including one row:

- `cublasLtMatmulDescCreate(COMPUTE_32F, CUDA_R_32F)`, then `TRANSA = OP_T`, `TRANSB = OP_N`,
  `EPILOGUE = EPILOGUE_BIAS`, `BIAS_POINTER = bias`
- layouts (column-major): A = weight `R_16F rows=in cols=out ld=in`, B = input
  `rows=in cols=M ld=in`, C = D = output `rows=out cols=M ld=out`, beta 0, C aliases D
- preference: `MAX_WORKSPACE_BYTES = 1048576` and `MIN_ALIGNMENT_{A,B,C,D}_BYTES` set
  to each pointer's alignment, largest power of two up to 256 (the log omits the
  default 256)
- first result of `cublasLtMatmulAlgoGetHeuristic`, workspace 1 MiB

## Linears without bias and the LM head: `cublasGemmEx`

The key projections (no bias) and `x @ embedding_T` run `cublasGemmEx` on a handle with
`CUBLAS_DEFAULT_MATH` and an 8519680-byte workspace (`cublasSetWorkspace_v2`), which cuBLAS
turns into `cublasLtHSHMatmul` with `smCountTarget=170`:

- keys: `transa=T transb=N m=1280 n=M k=1280 lda=ldb=ldc=1280`, all `CUDA_R_16F`,
  `CUBLAS_COMPUTE_32F`, `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, alpha 1, beta 0
- LM head: `transa=N transb=N m=51866 n=M k=1280 lda=51866 ldb=1280 ldc=51866` against
  `token_embedding.weight.T.contiguous()` (`TextDecoder.cache_embedding_weight`)

## Row counts that reach the GEMMs

Decode steps run inside CUDA graphs captured per bucket, so M is the bucket size
(1, 2, 4, 8, 16, 32, 42, 52), never the live row count. Encoder linears see
M = 1500 x windows in the prefill batch, the prefill M = 7 x windows, and the DTW
forward M = rows x padded tokens. The algorithm, and so the bits, change with M, so the
native runtime must form the same batches.

## Other calls seen

- ECAPA2 (TorchScript, FP32): cuBLASLt `COMPUTE_32F` with `EPILOGUE_BIAS` at M = 1 for
  its small linears, strided-batched `COMPUTE_32F_FAST_TF32` matmuls, and cuDNN backend
  graphs for its convolutions plus `cudnnBatchNormalizationForwardInference`
- DTW attention scores: 4 `cublasSgemmStridedBatched` calls (FP32, default math)
- encoder conv stem: cuDNN backend graphs (heuristic engine choice)

## Where the native runtime departs from torch: decode-step linears

At the decode buckets, torch's one-result, 1 MiB heuristic picks kernels that can't
stream the weights at DRAM rate. For example, the packed QKV at 32 to 54 rows runs one
160-block wave of 256x128 tiles in 36 µs, where the 9.8 MB of weights need 5.9 µs at
1.67 TB/s. Decode steps (one token per row) therefore run their linears with bias
through `Blas::linear_bias_decode` (`crates/whisperdrz-native/src/cuda/blas/decode.rs`).
It keeps torch's descriptors, compute type and bias epilogue but takes the algorithm
from a fixed table: the fastest deterministic candidate per (rows, shape) among up to 64
heuristic results under a 4 MiB workspace and their split-K variants with the FP32
(compute-type) reduction. A candidate qualifies only if its error against an exact f64
product stays within 1.25x torch's max-abs error and 1.05x its mean, and it must beat
torch's choice by 3%. That leaves mostly the 32-to-54-row buckets, where the picks are
both 2-4x faster than torch's kernels and more accurate. Below 32 rows, the split-K
candidates that would beat torch add about a third to the mean error, so those buckets
keep torch's choice. The table was measured by `tests/gpu_gemm_probe.rs` on the
RTX 5090 with cuBLAS 12.8.4 and is used only on that GPU (compute 12.0, 170 SMs). Those
GEMMs no longer match torch bit for bit; the noise-band gate judges them. Prefills, the
DTW forward, the encoder, the bias-free key projections and the LM head still make
torch's calls. To re-measure, run
`WHISPERDRZ_GEMM_PROBE=out.json cargo test --release --features cuda -p whisperdrz-native --test gpu_gemm_probe -- --nocapture`
through the GPU queue and paste its printed `pick(...)` lines into `TABLE`.
