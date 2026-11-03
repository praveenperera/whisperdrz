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
