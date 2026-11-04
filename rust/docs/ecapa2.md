# ECAPA2: the speaker embedder, and how the native runtime runs it

The speaker remap (`remap_speakers_per_line` in `transcriber.py`) embeds each line's audio
with ECAPA2 (`Jenthe/ECAPA2`, the TorchScript `ecapa2.pt`) through
`_embed_audio_segment`: raw 16 kHz mono FP32 audio `(1, T)` in, `model(x)` (labels
`"embedding"`), then `F.normalize` on the CPU. The model runs in FP32; the code notes that
half precision produces garbage.

`rust/tools/ecapa2.py` re-implements the TorchScript forward functionally. On the CPU it
equals the TorchScript output bit for bit on recorded segments (`ecapa2.py check`), which
is the proof that the description below is complete. `ecapa2.py export DIR` writes all
625 parameters and buffers as FP32 to `DIR/ecapa2.safetensors` under their TorchScript
names, with the checkpoint's sha256 in the metadata. The native port is
`crates/whisperdrz-native/src/cuda/ecapa2.rs` (host) and `kernels/src/ecapa2.rs`
(cuda-oxide glue kernels).

## Architecture

Shapes are for one input of `T` samples; `F = 1 + (T - 1) / 160` frames. Every batch
norm is inference-mode with eps 1e-5, `(x - mean) / sqrt(var + eps) * weight + bias`.

### Front end (`pre_processing`)

1. `torchaudio.functional.spectrogram`: `n_fft` 511, `win_length` 400, hop 160, power 2,
   centered with reflect padding (255 samples each side), one-sided: 256 bins. The
   window buffer `pre_processing.0.window` is a 400-point Hamming window (0.08 at both
   ends); torch pads it with zeros to 511, centered at offset 55. Output `(256, F)`
2. `log(x + 1e-6)`
3. Subtract each bin's mean over time

### 2D trunk

The spectrogram becomes `(1, 1, 256, F)`: one channel, frequency as height, time as
width. Each block is a `Sequential`:

```text
conv3x3 (padding 1, bias) -> ReLU -> BN2d     indices 0, 1, 2
conv3x3 (padding 1, bias) -> ReLU -> BN2d     indices 3, 4, 5
conv3x3 (padding 1, bias) -> ReLU -> BN2d     indices 6, 7, 8
squeeze-excitation (index 9)
```

Squeeze-excitation: average over height and time per channel, then
`Linear(C, 128) -> ReLU -> BN1d -> Linear(128, C) -> sigmoid`, and the block output is
multiplied by that per-channel scale. A residual is added after it.

| Block | Channels | Height | First conv stride | Residual |
|---|---|---|---|---|
| conv_1 | 1 -> 164 | 256 | 1 | none |
| conv_2, conv_17 | 164 | 256 | 1 | identity |
| conv_3 | 164 | 256 -> 128 | (2, 1) | downsample_1 |
| conv_4, conv_5, conv_18 | 164 | 128 | 1 | identity |
| conv_6 | 164 | 128 -> 64 | (2, 1) | downsample_2 |
| conv_7 | 164 | 64 | 1 | identity |
| conv_8 | 164 -> 192 | 64 | 1 | none |
| conv_19 | 192 | 64 | 1 | identity |
| conv_9 | 192 | 64 -> 32 | (2, 1) | downsample_3 |
| conv_10, conv_11, conv_20 | 192 | 32 | 1 | identity |
| conv_12 | 192 | 32 -> 16 | (2, 1) | downsample_4 |
| conv_13 to conv_16 | 192 | 16 | 1 | identity |

The order is the forward's, not the numeric one. A downsample is a 1x1 conv with stride
(2, 1), no padding and no bias, then BN2d. Strides only shrink frequency; time keeps all
`F` frames throughout.

### 1D head

The trunk output `(1, 192, 16, F)` is reshaped to `(1, 3072, F)` with channel index
`c * 16 + h`.

1. `tdnn_1`: conv1d 3072 -> 1024 (kernel 1, bias) -> ReLU -> BN1d
2. `tdnn_2`, added to the `tdnn_1` output as a residual:
   1. conv1d 1024 -> 1024 (k 1) -> ReLU -> BN1d
   2. Res2Net, 8 groups of 128 channels: group 0 passes through. Group 1 is
      `BN(ReLU(conv(pad(x_1))))`, and group i >= 2 is `BN(ReLU(conv(pad(x_i + y_{i-1}))))`.
      Each conv is 128 -> 128 with kernel 3, dilation 2 and bias, after reflect padding of
      2 frames per side
   3. conv1d 1024 -> 1024 (k 1) -> ReLU -> BN1d
   4. squeeze-excitation over time: the mean over frames goes through the same MLP shape
      as the trunk's (1024 -> 128 -> 1024), and the result scales each channel
3. `dense_1`: conv1d 1024 -> 1536 (k 1) -> ReLU, without a batch norm

### Attentive statistics pooling (`pooling_1`)

1. Global context: `[x; mean_t(x); sqrt(var_t(x) + 1e-5)]` with the unbiased variance,
   broadcast over time: `(4608, F)`
2. Attention: conv1d 4608 -> 128 -> ReLU -> BN1d -> conv1d 128 -> 1536 (each k 1,
   bias), then a softmax over time per channel gives the weights `w`
3. `mu = sum_t(w x)` and `sigma = sqrt(|sum_t(w x * x) - mu^2| + 1e-5)`
4. `cat(mu, sigma)` gives 3072 values, then BN1d (`pooling_1.1`)

### Output

`dense_2`: Linear 3072 -> 192 with bias. The model returns this unnormalized; the caller
applies `F.normalize` (`e / max(|e|, 1e-12)`).

## What torch dispatches

Captured with the CUDA profiler on one 99,520-sample segment (torch 2.8.0+cu128, cuDNN
9.10.2, RTX 5090; written by `ecapa2.py ref` to `dispatch.txt`). The transcriber never
sets the TF32 flags, so torch's defaults hold: `torch.backends.cudnn.allow_tf32 = True`
and `torch.backends.cuda.matmul.allow_tf32 = False`.

| Op | Kernel | Precision |
|---|---|---|
| STFT | cuFFT (radix 7 x prime 73, packR2C) | FP32 |
| conv3x3 with C >= 164 | `cutlass_tensorop_s1688fprop_optimized_tf32` (NHWC, with layout transposes) | TF32 |
| conv_1.0 (1 input channel), the 4 downsamples | `implicit_convolve_sgemm` | FP32 |
| Res2Net convs | `sm80_xmma_fprop_implicit_gemm_tf32f32` | TF32 |
| 1x1 conv1d (tdnn, dense_1, attention) | `cutlass_80_tensorop_s1688gemm` via cuDNN | TF32 |
| batch norm | `cudnn::bn_fw_inf`, and ATen's channels-last transform for BN1d | FP32 |
| squeeze-excitation and `dense_2` linears | cuBLAS gemv, dot | FP32 |
| softmax, reductions, elementwise | ATen kernels and TorchScript fusions | FP32 |

So the transcriber's embeddings carry TF32 rounding from most convolutions. The FP32
truth used by the tests runs the same graph with `cudnn.allow_tf32 = False`.

## The native port

Activations are FP32. cuBLAS carries every convolution, the DFT and `dense_2`
(`Blas::gemm_f32`, `cublasGemmStridedBatchedEx` with `CUBLAS_COMPUTE_32F`,
`CUBLAS_COMPUTE_32F_FAST_TF32` for the TF32 convs, or FP16 A and B with
`CUBLAS_COMPUTE_32F` for the FP16-input convs). The `ecapa_*` cuda-oxide kernels do the
rest.

### Layouts

- Trunk activations are `[rows][channels][cols]`: frequency rows, time innermost, with a
  one-element zero border on rows and columns (`Grid::trunk`). `cols` (`Wp`) is the
  frame count plus the border, rounded up to a multiple of 4 so every row starts 16-byte
  aligned; the extra columns are border too. Each kernel that writes a trunk buffer
  covers its whole extent and writes the border as zeros, so the border is always the
  next conv's zero padding.
- A 3x3 conv is one strided-batched GEMM per chunk of output rows, computing all three
  kernel columns `kw` at once over the unshifted input. In column-major terms it has
  `m = Wp` (every input column), `n = 3 * Co` and `k = 3 * Ci`, batched over the output
  rows. The A operand is the input from column 0 with `lda = Wp`; with that layout the
  three kernel rows and the input channels form one contiguous inner index
  `kh * Ci + ci` of stride `Wp`. Weights are re-laid on load to `[kw][Co][kh * Ci + ci]`
  (`conv3x3_layout`), which is the `[3 Co][3 Ci]` B operand. The product goes to a
  scratch buffer `[rows][kw][Co][Wp]`, and `ecapa_conv3x3_epilogue` sums the three
  kernel columns at their shifts (`kw` from column `w + kw - 1`), in the order three
  accumulating GEMMs would, then adds bias, ReLU and batch norm. A stride of 2 only
  changes the batch stride. The scratch holds at most 32 MiB, so the product of a long
  crop runs in row chunks that stay in L2 for the epilogue.
- Why one stacked GEMM: shifting the A pointer by `kw` columns, as three per-column
  GEMMs would, breaks 16-byte alignment and makes cuBLAS take its `align1` kernels.
  More of the loss, though, came from `n = Co = 164` filling 128-wide tiles only 64%;
  `n = 3 * Co = 492` fills them 96%. Measured on the RTX 5090 at one 128-row, 1000-frame
  conv: 1099 us for the three accumulating `align1` GEMMs, 1053 us with only the
  pointers aligned, 703 us for the stacked `align4` GEMM.
- A downsample is one strided-batched GEMM over every other input row, from column 0 so
  it stays aligned.
- `tdnn_1` reads the trunk output in place: interior rows and channels form one inner
  dimension of stride `Wp`. Its weight columns are reordered on load from torch's
  `c * 16 + h` to the buffer's `h * 192 + c`.
- Head activations are `[channels][frames]`. A 1x1 conv1d is one GEMM. Each Res2Net conv
  is three accumulating GEMMs over the reflect-padded group input at offsets 0, 2 and 4,
  with weights re-laid to `[tap][Co][Ci]`.
- The DFT is one GEMM of the `[F][511]` frames against a `[511][512]` cos and -sin basis.
  The basis is built in f64 with the angle reduced mod 511, then rounded to f32. The
  real parts land in columns 0..256 and the imaginary parts in 256..512.
- Pooling writes `(mu, sigma)` pairs per channel. `dense_2`'s weight columns are
  interleaved to match on load. `dense_2` is a transposed GEMM onto a copy of its bias,
  in full FP32 in both modes, as torch runs it.
- `F.normalize` runs on the host in f64 (`normalize`).

### Batch norm

Each batch norm is applied in its producer's epilogue, as torch computes it:
`(x - mean) * k + b`, with `k = weight / sqrt(var + eps)` computed in f64 on load. It is
not folded into the convolution weights, for these reasons:

- Every conv batch norm here follows a ReLU, so it can't fold into the conv before it.
- Folding it into the next conv is exact only away from the zero padding. The padding
  zeros come after the batch norm, so at the borders the fold would read
  `-mean * k + b` where torch reads 0. It would also change the identity residual.
- Only the four downsamples (conv, then BN) could fold exactly. That would save four
  elementwise passes, and the residual add still needs the bias.

There is no folded variant to compare against: the fold is either inexact at every
border or saves almost nothing.

### Precision modes (`Precision`)

- `Fp32`: every GEMM in full FP32. This is the closest to the FP32 truth.
- `Reference`: TF32 where torch uses it (every conv except conv_1.0 and the
  downsamples), full FP32 for the DFT and `dense_2`. This mirrors the transcriber's
  dispatch, but cuBLAS's TF32 kernels are not cuDNN's, so the rounding differs in
  detail.
- `F16Convs`, what the runtime uses: `Reference`, with the trunk's TF32 3x3 convs
  taking FP16 inputs and summing in FP32. FP16 keeps TF32's 10-bit mantissa and runs at
  twice the tensor rate; the trunk's values stay far inside its exponent range. Each
  block's first conv reads an FP16 copy of the block input, written by the previous
  block's `ecapa_scale_residual_f16` beside the FP32 one the residual reads. The first
  two convs write only FP16 (`ecapa_conv3x3_epilogue_f16`) and the third writes FP32
  for the squeeze-excitation. Trunk rows round to 8 columns instead of 4, and the FP16
  weight rows pad to 8 values (`conv3x3_layout_f16`), so every FP16 row starts 16-byte
  aligned. cuBLAS runs an `align8` kernel for the 192-channel blocks but `align2` ones for
  the 164-channel blocks, whose `k = 3 * 164 = 492` is not a multiple of 8. conv_1.0,
  the downsamples and the head keep their `Reference` precision.

## Tests

`scripts/gpu-test.sh` runs these through the GPU queue (`tests/gpu_ecapa2.rs`):

- `kernels_match_f64`: every glue kernel and conv GEMM (both strides, the downsample, a
  strided conv1d, the DFT) against an f64 CPU reference on random inputs. Outputs start
  as NaN, so a skipped element fails.
- `layers_match_torch_fp32`: each block's output on three recorded segments (100, 546
  and 996 frames) against torch's FP32 truth, beside torch's own error in the
  transcriber's mode. A layer passes when our max-abs and mean-abs errors are both at
  most 1.5x torch's. The spectrogram runs on cuFFT in both torch modes, so torch's
  reference-mode error there is zero by construction. For that tap, ours and torch's are
  both measured against an f64 spectrogram computed on the host.
- `embeddings_match_recorded`: all 76 recorded `_embed_audio_segment` calls (short3 and
  e21c600) end to end, as cosine similarity against the recorded and the truth
  embeddings. It writes our embeddings to `ours.json` for the remap replay.

`tools/ecapa2_remap_check.py ours.json` (CPU) replays `remap_speakers_per_line` on each
short3 file with our embeddings and the recorded random window draws, and checks the
output text against the recorded one. It also replays the recorded embeddings (a control
that must reproduce the trace) and torch's FP32 truth.

## Results

Torch 2.8.0+cu128, RTX 5090.

Kernels against f64: all within bound. The largest error is the DFT GEMM, 3.7e-6 max-abs
over a frame with an L1 norm of about 100.

Layers in `Fp32` mode: all 78 rows (3 segments x 26 taps) pass. The worst ratios of our
error to torch's are 0.78 on max-abs (e21c600 conv_2) and 0.39 on mean-abs (ikgcq
spec). Past the first blocks our mean-abs error is about 10x to 100x below torch's
reference-mode error.

Layers in `Reference` mode: 77 of 78 rows within 1.5x. On ikgcq conv_10 the max-abs ratio
is 1.71 (9.24e-3 against 5.39e-3), while the mean-abs ratio is 0.997. The two modes run
different TF32 kernels, which round differently in individual elements. This mode is
not the one the runtime uses.

End to end, cosine over 76 segments:

| Mode | vs recorded (min / mean) | vs FP32 truth (min / mean) |
|---|---|---|
| `Fp32` | 0.999997526 / 0.999999725 | 1.000000000 / 1.000000000 |
| `Reference` | 0.999997349 / 0.999999821 | 0.999999624 / 0.999999870 |
| torch FP32 truth | 0.9999975 (min) | |

Layers in `F16Convs` mode: all 78 rows within 1.5x of torch's reference-mode error; the
worst max-abs ratio is 1.39 (e21c600 conv_3) and the worst mean-abs ratio 1.05
(e21c600 embedding). End to
end, cosine over the 76 segments is 0.999996995 min / 0.999999821 mean against the
recorded embeddings and 0.999999510 / 0.999999865 against the FP32 truth.

Speaker remap on short3 (nctdh, willh, ikgcq): the output is identical to the trace in
all five replays (recorded, `Fp32`, `Reference`, `F16Convs`, torch truth).

On ikgcq one recorded pairwise distance is 6.0e-4 from the 1.1 threshold. Our `Fp32`
embeddings move pairwise distances by up to 5.39e-4, the same amount torch's own FP32
truth moves them; `F16Convs` moves them by up to 5.08e-4. The margin is torch's TF32-vs-FP32 difference, not the port's.
