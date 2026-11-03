//! ECAPA2 (`Jenthe/ECAPA2`), the speaker embedder the transcriber's speaker remap runs
//! on each line's audio, in FP32 on the GPU
//!
//! `rust/docs/ecapa2.md` has the architecture and how each step maps to torch's. In short:
//! a 511-point spectrogram, log and per-bin centering; a 2D trunk of 20 residual blocks of
//! three 3x3 conv + ReLU + batch norm layers with squeeze-excitation; a 1D head (1x1
//! conv, a Res2Net block with squeeze-excitation, a 1x1 conv); attentive statistics
//! pooling; a linear layer to 192 values. The weights come from
//! `rust/tools/ecapa2.py export` as FP32 safetensors under their TorchScript names.
//!
//! Every convolution, the DFT and the last linear layer are cuBLAS FP32 GEMMs; the glue
//! is the `ecapa_*` cuda-oxide kernels. The 2D trunk stores activations as
//! `[rows][channels][cols]` with frequency as rows, time as columns and a one-element
//! zero border, so a 3x3 conv is three strided-batched GEMMs over shifted views (one per
//! kernel column, with the three kernel rows folded into the inner dimension) and the
//! trunk output feeds the head's first 1x1 conv in place

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use eyre::{Result, WrapErr, ensure, eyre};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use super::blas::{Blas, F32Math, Operand};
use super::device::{Arg, Buf, Gpu, blocks};

pub const SAMPLE_RATE: usize = 16_000;
/// Embedding width
pub const DIM: usize = 192;
const N_FFT: usize = 511;
const WIN_LENGTH: usize = 400;
const HOP: usize = 160;
/// One-sided bins of the 511-point transform
const BINS: usize = N_FFT / 2 + 1;
/// torch.stft centers a window shorter than the transform in it
const WIN_OFFSET: usize = (N_FFT - WIN_LENGTH) / 2;
const BN_EPS: f64 = 1e-5;
/// Hidden width of every squeeze-excitation MLP, the kernels' `SE_HIDDEN`
const SE_HIDDEN: usize = 128;
const RES2_GROUPS: usize = 8;
const RES2_WIDTH: usize = 128;
const RES2_DILATION: usize = 2;
const HEAD: usize = 1024;
const POOL_IN: usize = 1536;
const ATTENTION: usize = 128;
const ELEMENTWISE_BLOCK: u32 = 256;
const ROW_BLOCK: u32 = 256;

/// The 2D trunk in order: block name, stride of its first conv along frequency, and the
/// residual added after squeeze-excitation
const TRUNK: [(&str, usize, Residual); 20] = [
    ("conv_1", 1, Residual::None),
    ("conv_2", 1, Residual::Identity),
    ("conv_17", 1, Residual::Identity),
    ("conv_3", 2, Residual::Downsample("downsample_1")),
    ("conv_4", 1, Residual::Identity),
    ("conv_5", 1, Residual::Identity),
    ("conv_18", 1, Residual::Identity),
    ("conv_6", 2, Residual::Downsample("downsample_2")),
    ("conv_7", 1, Residual::Identity),
    ("conv_8", 1, Residual::None),
    ("conv_19", 1, Residual::Identity),
    ("conv_9", 2, Residual::Downsample("downsample_3")),
    ("conv_10", 1, Residual::Identity),
    ("conv_11", 1, Residual::Identity),
    ("conv_20", 1, Residual::Identity),
    ("conv_12", 2, Residual::Downsample("downsample_4")),
    ("conv_13", 1, Residual::Identity),
    ("conv_14", 1, Residual::Identity),
    ("conv_15", 1, Residual::Identity),
    ("conv_16", 1, Residual::Identity),
];

#[derive(Clone, Copy, Debug)]
enum Residual {
    /// `x + block(x)`
    Identity,
    /// `block(x) + bn(conv1x1_stride2(x))`
    Downsample(&'static str),
    /// No residual: the block changes the channel count
    None,
}

/// GEMM precision per layer kind
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// Every GEMM in full FP32: closest to the FP32 truth
    Fp32,
    /// TF32 where the reference's cuDNN convolutions use TF32 (every conv but the first
    /// and the 1x1 downsamples), full FP32 for the DFT and the linear layer, as torch runs
    /// ECAPA2 under its default flags
    Reference,
}

/// Shape of an activation buffer: `[rows][channels][cols]` with a `pad`-wide zero border
/// on rows and cols. The 2D trunk uses pad 1 (rows are frequency, cols time); the 1D head
/// uses one row and pad 0
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    pub rows: usize,
    pub channels: usize,
    pub cols: usize,
    pub pad: usize,
}

impl Grid {
    /// A trunk buffer for `height` frequency rows of `channels` over `frames`
    pub fn trunk(height: usize, channels: usize, frames: usize) -> Self {
        Self {
            rows: height + 2,
            channels,
            cols: frames + 2,
            pad: 1,
        }
    }

    /// A head buffer `[channels][frames]`
    pub fn head(channels: usize, frames: usize) -> Self {
        Self {
            rows: 1,
            channels,
            cols: frames,
            pad: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.rows * self.channels * self.cols
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Interior rows and columns
    pub fn height(&self) -> usize {
        self.rows - 2 * self.pad
    }

    pub fn width(&self) -> usize {
        self.cols - 2 * self.pad
    }

    /// Element offset of interior row 0, channel 0, column 0
    fn origin(&self) -> usize {
        (self.pad * self.channels) * self.cols + self.pad
    }

    fn args(&self) -> [Arg; 4] {
        [
            self.rows.into(),
            self.channels.into(),
            self.cols.into(),
            self.pad.into(),
        ]
    }

    /// The interior as torch's `[channels][height][width]` (or `[channels][width]`)
    pub fn to_torch(&self, data: &[f32]) -> Vec<f32> {
        let (h, w) = (self.height(), self.width());
        let mut out = Vec::with_capacity(self.channels * h * w);
        for c in 0..self.channels {
            for r in 0..h {
                let start = ((r + self.pad) * self.channels + c) * self.cols + self.pad;
                out.extend_from_slice(&data[start..start + w]);
            }
        }

        out
    }
}

/// Address of element `offset` of an FP32 buffer
fn at(buf: &Buf, offset: usize) -> u64 {
    buf.at(offset * size_of::<f32>())
}

/// A kernel slice argument of `len` FP32 values from element `offset`
fn slice(buf: &Buf, offset: usize, len: usize) -> Arg {
    debug_assert!((offset + len) * size_of::<f32>() <= buf.bytes());
    Arg::slice(at(buf, offset), len)
}

/// The steps of the forward pass, each one kernel or GEMM sequence on the GPU's stream.
/// Public so the GPU tests can check each against an f64 CPU reference
pub struct Steps<'a> {
    pub gpu: &'a Gpu,
    pub blas: &'a Blas,
}

impl Steps<'_> {
    fn elementwise(&self, name: &'static str, n: usize, args: &[Arg]) -> Result<()> {
        self.gpu.launch(
            name,
            (blocks(n, ELEMENTWISE_BLOCK), 1, 1),
            (ELEMENTWISE_BLOCK, 1, 1),
            0,
            args,
        )
    }

    /// One block per row of a `[rows][width]` output
    fn per_row(&self, name: &'static str, rows: usize, args: &[Arg]) -> Result<()> {
        self.gpu.launch(
            name,
            (u32::try_from(rows)?, 1, 1),
            (ROW_BLOCK, 1, 1),
            0,
            args,
        )
    }

    /// `[frames][N_FFT]` windowed frames of `audio` (`samples` long), centered with
    /// reflect padding
    pub fn frames(&self, out: &Buf, audio: &Buf, samples: usize, window: &Buf) -> Result<usize> {
        let frames = 1 + (samples - 1) / HOP;
        ensure!(out.bytes() >= frames * N_FFT * 4, "frame buffer too small");
        self.elementwise(
            "ecapa_frames",
            frames * N_FFT,
            &[
                slice(out, 0, frames * N_FFT),
                slice(audio, 0, samples),
                slice(window, 0, WIN_LENGTH),
                N_FFT.into(),
                HOP.into(),
                WIN_OFFSET.into(),
            ],
        )?;
        Ok(frames)
    }

    /// The one-sided DFT of `frames` rows as one FP32 GEMM against the `[N_FFT][2 BINS]`
    /// cos and -sin basis: `[frames][2 BINS]`, real parts then imaginary parts
    pub fn dft(&self, out: &Buf, frames_buf: &Buf, basis: &Buf, frames: usize) -> Result<()> {
        self.blas.gemm_f32(
            F32Math::Full,
            (false, false),
            (2 * BINS, frames, N_FFT),
            Operand::new(basis.ptr(), 2 * BINS),
            Operand::new(frames_buf.ptr(), N_FFT),
            Operand::new(out.ptr(), 2 * BINS),
            1,
            0.0,
        )
    }

    /// `log(|X|^2 + 1e-6)` as `[BINS][frames]`
    pub fn log_power(&self, out: &Buf, spectrum: &Buf, frames: usize) -> Result<()> {
        self.elementwise(
            "ecapa_log_power",
            BINS * frames,
            &[
                slice(out, 0, BINS * frames),
                slice(spectrum, 0, 2 * BINS * frames),
                frames.into(),
                BINS.into(),
            ],
        )
    }

    /// Each row of `x` (`[height][frames]`) minus its mean over time, into the trunk
    /// input `out` with one channel
    pub fn center_rows(&self, out: &Buf, x: &Buf, height: usize, frames: usize) -> Result<Grid> {
        let grid = Grid::trunk(height, 1, frames);
        self.per_row(
            "ecapa_center_rows",
            grid.rows,
            &[
                slice(out, 0, grid.len()),
                grid.cols.into(),
                slice(x, 0, height * frames),
                grid.rows.into(),
                frames.into(),
                grid.pad.into(),
            ],
        )?;
        Ok(grid)
    }

    /// 3x3 conv (padding 1, stride `stride` along rows) of trunk buffer `x` into `out`,
    /// without bias: one strided-batched GEMM per kernel column over the output rows.
    /// `w` is `[3 kw][co][3 kh * ci]`
    pub fn conv3x3(
        &self,
        math: F32Math,
        (x, g): (&Buf, Grid),
        out: &Buf,
        w: &Buf,
        co: usize,
        stride: usize,
    ) -> Result<Grid> {
        let ci = g.channels;
        let height = g.height() / stride;
        let o = Grid::trunk(height, co, g.width());
        ensure!(
            g.pad == 1 && g.height().is_multiple_of(stride),
            "conv3x3 input {g:?}"
        );
        ensure!(o.len() * 4 <= out.bytes(), "conv3x3 output {o:?} overflows");
        for kw in 0..3 {
            self.blas.gemm_f32(
                math,
                (false, false),
                (o.width(), co, 3 * ci),
                Operand::strided(at(x, kw), g.cols, stride * ci * g.cols),
                Operand::new(at(w, kw * co * 3 * ci), 3 * ci),
                Operand::strided(at(out, o.origin()), o.cols, co * o.cols),
                height,
                if kw == 0 { 0.0 } else { 1.0 },
            )?;
        }

        Ok(o)
    }

    /// 1x1 conv with stride 2 along rows and no padding (the downsample path), without
    /// bias; `w` is `[co][ci]`
    pub fn conv1x1_s2(
        &self,
        math: F32Math,
        x: &Buf,
        g: Grid,
        out: &Buf,
        w: &Buf,
        co: usize,
    ) -> Result<Grid> {
        let ci = g.channels;
        let o = Grid::trunk(g.height() / 2, co, g.width());
        ensure!(
            g.pad == 1 && g.height().is_multiple_of(2),
            "downsample input {g:?}"
        );
        ensure!(
            o.len() * 4 <= out.bytes(),
            "downsample output {o:?} overflows"
        );
        self.blas.gemm_f32(
            math,
            (false, false),
            (o.width(), co, ci),
            Operand::strided(at(x, g.origin()), g.cols, 2 * ci * g.cols),
            Operand::new(w.ptr(), ci),
            Operand::strided(at(out, o.origin()), o.cols, co * o.cols),
            o.height(),
            0.0,
        )?;
        Ok(o)
    }

    /// `y[co][t] = sum_ci w[co][ci] x[ci][t]` for head buffers, `beta` scaling what `y`
    /// held; the input `(x, ld)` may be any `[ci][frames]` view with row stride `ld`
    pub fn conv1d(
        &self,
        math: F32Math,
        (x, ld): (u64, usize),
        y: u64,
        w: u64,
        (ci, co, frames): (usize, usize, usize),
        beta: f32,
    ) -> Result<()> {
        self.blas.gemm_f32(
            math,
            (false, false),
            (frames, co, ci),
            Operand::new(x, ld),
            Operand::new(w, ci),
            Operand::new(y, frames),
            1,
            beta,
        )
    }

    /// Bias, ReLU and batch norm in place over buffer `y` of shape `g`, zeroing its
    /// border; empty `bias` or `bn` skip that step
    pub fn epilogue(
        &self,
        y: &Buf,
        offset: usize,
        g: Grid,
        bias: Arg,
        bn: Arg,
        relu: bool,
    ) -> Result<()> {
        let [rows, channels, cols, pad] = g.args();
        self.elementwise(
            "ecapa_conv_epilogue",
            g.len(),
            &[
                slice(y, offset, g.len()),
                bias,
                bn,
                rows,
                channels,
                cols,
                pad,
                u32::from(relu).into(),
            ],
        )
    }

    /// Per-channel mean of the interior of `x`
    pub fn channel_mean(&self, out: &Buf, x: &Buf, g: Grid) -> Result<()> {
        let [rows, channels, cols, pad] = g.args();
        self.per_row(
            "ecapa_channel_mean",
            g.channels,
            &[
                slice(out, 0, g.channels),
                1u32.into(),
                slice(x, 0, g.len()),
                rows,
                channels,
                cols,
                pad,
            ],
        )
    }

    /// Squeeze-excitation's MLP on a channel mean: the per-channel scales
    pub fn se_fc(&self, out: &Buf, mean: &Buf, se: &Se) -> Result<()> {
        let c = se.channels;
        self.per_row(
            "ecapa_se_fc",
            1,
            &[
                slice(out, 0, c),
                c.into(),
                slice(mean, 0, c),
                slice(&se.w1, 0, SE_HIDDEN * c),
                slice(&se.b1, 0, SE_HIDDEN),
                slice(&se.bn, 0, 3 * SE_HIDDEN),
                slice(&se.w2, 0, c * SE_HIDDEN),
                slice(&se.b2, 0, c),
            ],
        )
    }

    /// `out = x * s[channel] + r` (no `r` for none) over buffers of shape `g`
    pub fn scale_residual(
        &self,
        out: &Buf,
        x: &Buf,
        s: &Buf,
        r: Option<&Buf>,
        g: Grid,
    ) -> Result<()> {
        let [rows, channels, cols, pad] = g.args();
        self.elementwise(
            "ecapa_scale_residual",
            g.len(),
            &[
                slice(out, 0, g.len()),
                slice(x, 0, g.len()),
                slice(s, 0, g.channels),
                r.map_or(Arg::EMPTY, |r| slice(r, 0, g.len())),
                rows,
                channels,
                cols,
                pad,
            ],
        )
    }

    /// `a + b` (or `a` alone) over `[channels][frames]`, reflect-padded by `pad` frames
    pub fn reflect_add(
        &self,
        out: &Buf,
        a: Arg,
        b: Arg,
        (channels, frames, pad): (usize, usize, usize),
    ) -> Result<()> {
        let n = channels * (frames + 2 * pad);
        self.elementwise(
            "ecapa_reflect_add",
            n,
            &[slice(out, 0, n), a, b, frames.into(), pad.into()],
        )
    }

    /// `[x; mean; std]` over time for `x` `[channels][frames]`
    pub fn stats_cat(&self, out: &Buf, x: &Buf, channels: usize, frames: usize) -> Result<()> {
        self.per_row(
            "ecapa_stats_cat",
            3 * channels,
            &[
                slice(out, 0, 3 * channels * frames),
                frames.into(),
                slice(x, 0, channels * frames),
                channels.into(),
            ],
        )
    }

    /// Attentive statistics pooling plus its batch norm: `[channels][2]` (mu, sigma)
    pub fn attentive_pool(
        &self,
        out: &Buf,
        logits: &Buf,
        x: &Buf,
        bn: &Buf,
        channels: usize,
        frames: usize,
    ) -> Result<()> {
        self.per_row(
            "ecapa_attentive_pool",
            channels,
            &[
                slice(out, 0, 2 * channels),
                2u32.into(),
                slice(logits, 0, channels * frames),
                slice(x, 0, channels * frames),
                slice(bn, 0, 6 * channels),
                frames.into(),
            ],
        )
    }
}

/// One conv layer of the trunk: weights `[3 kw][co][3 kh * ci]`, bias, batch norm
struct Conv2d {
    w: Buf,
    bias: Buf,
    bn: Buf,
    co: usize,
}

/// A squeeze-excitation MLP
pub struct Se {
    w1: Buf,
    b1: Buf,
    bn: Buf,
    w2: Buf,
    b2: Buf,
    channels: usize,
}

impl Se {
    /// Uploads an MLP over `channels`: `w1` `[SE_HIDDEN][channels]`, `bn` as
    /// `[mean; weight / sqrt(var + eps); bias]` over the hidden units, `w2`
    /// `[channels][SE_HIDDEN]`
    pub fn upload(
        gpu: &Gpu,
        w1: Vec<f32>,
        b1: Vec<f32>,
        bn: Vec<f32>,
        w2: Vec<f32>,
        b2: Vec<f32>,
    ) -> Result<Self> {
        let channels = b2.len();
        ensure!(
            w1.len() == SE_HIDDEN * channels
                && b1.len() == SE_HIDDEN
                && bn.len() == 3 * SE_HIDDEN
                && w2.len() == channels * SE_HIDDEN,
            "squeeze-excitation shapes for {channels} channels"
        );
        Ok(Self {
            w1: gpu.upload(w1)?,
            b1: gpu.upload(b1)?,
            bn: gpu.upload(bn)?,
            w2: gpu.upload(w2)?,
            b2: gpu.upload(b2)?,
            channels,
        })
    }
}

struct Block {
    convs: [Conv2d; 3],
    se: Se,
    stride: usize,
    /// Downsample weights `[co][ci]` and batch norm
    downsample: Option<(Buf, Buf)>,
    residual: Residual,
    name: &'static str,
}

/// A 1x1 conv1d (`w` `[co][ci]`), its bias and optional batch norm
struct Conv1 {
    w: Buf,
    bias: Buf,
    bn: Option<Buf>,
    ci: usize,
    co: usize,
}

/// One Res2Net group conv: weights `[3 taps][co][ci]`, bias, batch norm
struct Res2Conv {
    w: Buf,
    bias: Buf,
    bn: Buf,
}

/// An activation the forward pass reports, in torch's layout
pub struct Tap {
    pub name: &'static str,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// ECAPA2 on the GPU
pub struct Ecapa2 {
    window: Buf,
    basis: Buf,
    trunk: Vec<Block>,
    tdnn1: Conv1,
    tdnn2_in: Conv1,
    res2: Vec<Res2Conv>,
    tdnn2_out: Conv1,
    tdnn2_se: Se,
    dense1: Conv1,
    attention_in: Conv1,
    attention_out: Conv1,
    pool_bn: Buf,
    /// `[DIM][2 POOL_IN]` with columns interleaved as (mu, sigma) per channel, the order
    /// the pooling kernel writes
    dense2_w: Buf,
    dense2_b: Buf,
    precision: Precision,
}

/// FP32 tensors of the exported checkpoint, on the host
struct Host<'a>(HashMap<String, (Vec<usize>, &'a [u8])>);

impl Host<'_> {
    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let (have, bytes) = self
            .0
            .get(name)
            .ok_or_else(|| eyre!("ECAPA2 weights have no {name}"))?;
        ensure!(have == shape, "{name} has shape {have:?}, want {shape:?}");
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }

    /// Batch norm `prefix` as `[mean; weight / sqrt(var + eps); bias]`
    fn bn(&self, prefix: &str, c: usize) -> Result<Vec<f32>> {
        let mean = self.get(&format!("{prefix}.running_mean"), &[c])?;
        let var = self.get(&format!("{prefix}.running_var"), &[c])?;
        let weight = self.get(&format!("{prefix}.weight"), &[c])?;
        let bias = self.get(&format!("{prefix}.bias"), &[c])?;
        let k = weight
            .iter()
            .zip(&var)
            .map(|(&w, &v)| (w as f64 / (v as f64 + BN_EPS).sqrt()) as f32);
        Ok(mean.iter().copied().chain(k).chain(bias).collect())
    }
}

impl Ecapa2 {
    /// Loads `ecapa2.safetensors` from `rust/tools/ecapa2.py export`
    pub fn load(gpu: &Gpu, path: &Path, precision: Precision) -> Result<Self> {
        let file = File::open(path).wrap_err_with(|| format!("open {}", path.display()))?;
        // read-only, and nothing else in the process writes it
        let map =
            unsafe { Mmap::map(&file) }.wrap_err_with(|| format!("map {}", path.display()))?;
        let st = SafeTensors::deserialize(&map)
            .wrap_err_with(|| format!("read safetensors header of {}", path.display()))?;
        let mut tensors = HashMap::new();
        for (name, view) in st.tensors() {
            ensure!(
                view.dtype() == Dtype::F32,
                "{name} is {:?}, want F32",
                view.dtype()
            );
            tensors.insert(name, (view.shape().to_vec(), view.data()));
        }
        let host = Host(tensors);
        let up = |v: Vec<f32>| gpu.upload(v);

        let window = host.get("pre_processing.0.window", &[WIN_LENGTH])?;
        let mut trunk = Vec::with_capacity(TRUNK.len());
        let mut ci = 1;
        for (name, stride, residual) in TRUNK {
            let co = host
                .0
                .get(&format!("{name}.0.weight"))
                .map(|(s, _)| s[0])
                .ok_or_else(|| eyre!("no {name}.0.weight"))?;
            let mut convs = Vec::with_capacity(3);
            for (layer, layer_ci) in [(0, ci), (3, co), (6, co)] {
                let w = host.get(&format!("{name}.{layer}.weight"), &[co, layer_ci, 3, 3])?;
                convs.push(Conv2d {
                    w: up(conv3x3_layout(&w, co, layer_ci))?,
                    bias: up(host.get(&format!("{name}.{layer}.bias"), &[co])?)?,
                    bn: up(host.bn(&format!("{name}.{}", layer + 2), co)?)?,
                    co,
                });
            }
            let se = Se {
                w1: up(host.get(&format!("{name}.9.fc.0.weight"), &[SE_HIDDEN, co])?)?,
                b1: up(host.get(&format!("{name}.9.fc.0.bias"), &[SE_HIDDEN])?)?,
                bn: up(host.bn(&format!("{name}.9.fc.2"), SE_HIDDEN)?)?,
                w2: up(host.get(&format!("{name}.9.fc.3.weight"), &[co, SE_HIDDEN])?)?,
                b2: up(host.get(&format!("{name}.9.fc.3.bias"), &[co])?)?,
                channels: co,
            };
            let downsample = match residual {
                Residual::Downsample(ds) => Some((
                    up(host.get(&format!("{ds}.0.weight"), &[co, ci, 1, 1])?)?,
                    up(host.bn(&format!("{ds}.1"), co)?)?,
                )),
                _ => None,
            };
            let convs: [Conv2d; 3] = convs.try_into().map_err(|_| eyre!("three convs"))?;
            trunk.push(Block {
                convs,
                se,
                stride,
                downsample,
                residual,
                name,
            });
            ci = co;
        }

        // the trunk ends at 16 frequency rows of 192 channels, which torch flattens
        // channel-major (c * 16 + h); the trunk buffer is row-major, so the columns of
        // tdnn_1's weight are reordered to h * 192 + c
        let (rows, ch) = (16, ci);
        let tdnn1 = host.get("tdnn_1.0.weight", &[HEAD, ch * rows, 1])?;
        let mut tdnn1_w = vec![0f32; tdnn1.len()];
        for o in 0..HEAD {
            for c in 0..ch {
                for h in 0..rows {
                    tdnn1_w[o * ch * rows + h * ch + c] = tdnn1[o * ch * rows + c * rows + h];
                }
            }
        }
        let conv1 = |prefix: &str,
                     bn: Option<&str>,
                     ci: usize,
                     co: usize,
                     w: Option<Vec<f32>>|
         -> Result<Conv1> {
            let w = match w {
                Some(w) => w,
                None => host.get(&format!("{prefix}.weight"), &[co, ci, 1])?,
            };
            Ok(Conv1 {
                w: up(w)?,
                bias: up(host.get(&format!("{prefix}.bias"), &[co])?)?,
                bn: bn.map(|b| host.bn(b, co).and_then(up)).transpose()?,
                ci,
                co,
            })
        };

        let mut res2 = Vec::with_capacity(RES2_GROUPS - 1);
        for i in 0..RES2_GROUPS - 1 {
            let w = host.get(
                &format!("tdnn_2.3.conv_filters.{i}.weight"),
                &[RES2_WIDTH, RES2_WIDTH, 3],
            )?;
            // [co][ci][tap] to [tap][co][ci]
            let mut taps = vec![0f32; w.len()];
            for o in 0..RES2_WIDTH {
                for c in 0..RES2_WIDTH {
                    for k in 0..3 {
                        taps[(k * RES2_WIDTH + o) * RES2_WIDTH + c] =
                            w[(o * RES2_WIDTH + c) * 3 + k];
                    }
                }
            }
            res2.push(Res2Conv {
                w: up(taps)?,
                bias: up(host.get(&format!("tdnn_2.3.conv_filters.{i}.bias"), &[RES2_WIDTH])?)?,
                bn: up(host.bn(&format!("tdnn_2.3.batch_norms.{i}"), RES2_WIDTH)?)?,
            });
        }

        // pooling writes (mu, sigma) pairs per channel; torch concatenates all mu, then
        // all sigma, so dense_2's columns are interleaved to match
        let dense2 = host.get("dense_2.0.weight", &[DIM, 2 * POOL_IN])?;
        let mut dense2_w = vec![0f32; dense2.len()];
        for j in 0..DIM {
            for c in 0..POOL_IN {
                dense2_w[j * 2 * POOL_IN + 2 * c] = dense2[j * 2 * POOL_IN + c];
                dense2_w[j * 2 * POOL_IN + 2 * c + 1] = dense2[j * 2 * POOL_IN + POOL_IN + c];
            }
        }

        let this = Self {
            window: up(window)?,
            basis: up(dft_basis())?,
            trunk,
            tdnn1: conv1("tdnn_1.0", Some("tdnn_1.2"), ch * rows, HEAD, Some(tdnn1_w))?,
            tdnn2_in: conv1("tdnn_2.0", Some("tdnn_2.2"), HEAD, HEAD, None)?,
            res2,
            tdnn2_out: conv1("tdnn_2.4", Some("tdnn_2.6"), HEAD, HEAD, None)?,
            tdnn2_se: Se {
                w1: up(host.get("tdnn_2.7.fc.0.weight", &[SE_HIDDEN, HEAD])?)?,
                b1: up(host.get("tdnn_2.7.fc.0.bias", &[SE_HIDDEN])?)?,
                bn: up(host.bn("tdnn_2.7.fc.2", SE_HIDDEN)?)?,
                w2: up(host.get("tdnn_2.7.fc.3.weight", &[HEAD, SE_HIDDEN])?)?,
                b2: up(host.get("tdnn_2.7.fc.3.bias", &[HEAD])?)?,
                channels: HEAD,
            },
            dense1: conv1("dense_1.0", None, HEAD, POOL_IN, None)?,
            attention_in: conv1(
                "pooling_1.0.attention.0",
                Some("pooling_1.0.attention.2"),
                3 * POOL_IN,
                ATTENTION,
                None,
            )?,
            attention_out: conv1("pooling_1.0.attention.3", None, ATTENTION, POOL_IN, None)?,
            pool_bn: up(host.bn("pooling_1.1", 2 * POOL_IN)?)?,
            dense2_w: up(dense2_w)?,
            dense2_b: up(host.get("dense_2.0.bias", &[DIM])?)?,
            precision,
        };
        gpu.synchronize()?;
        Ok(this)
    }

    /// Math for a conv GEMM; `exact` marks the convs the reference runs in FP32 even
    /// with TF32 allowed (its first conv and the 1x1 downsamples)
    fn conv_math(&self, exact: bool) -> F32Math {
        match self.precision {
            Precision::Reference if !exact => F32Math::Tf32,
            _ => F32Math::Full,
        }
    }

    /// The unnormalized embedding of 16 kHz mono `audio`, ECAPA2's `forward(x)`
    pub fn embed(&self, gpu: &Gpu, blas: &Blas, audio: &[f32]) -> Result<Vec<f32>> {
        self.forward(gpu, blas, audio, None)
    }

    /// The forward pass; `taps` collects each block's output in torch's layout
    pub fn forward(
        &self,
        gpu: &Gpu,
        blas: &Blas,
        audio: &[f32],
        mut taps: Option<&mut Vec<Tap>>,
    ) -> Result<Vec<f32>> {
        // torch's reflect padding needs more samples than the pad on each side
        ensure!(
            audio.len() > N_FFT / 2,
            "ECAPA2 needs more than {} samples, got {}",
            N_FFT / 2,
            audio.len()
        );
        let steps = Steps { gpu, blas };
        let mut tap = |name: &'static str, buf: &Buf, g: Grid| -> Result<()> {
            if let Some(taps) = taps.as_deref_mut() {
                let host: Vec<f32> = gpu.download(buf)?;
                let data = g.to_torch(&host[..g.len()]);
                let shape = match (g.pad, g.height()) {
                    (0, _) => vec![g.channels, g.width()],
                    (_, h) if g.channels == 1 => vec![h, g.width()],
                    (_, h) => vec![g.channels, h, g.width()],
                };
                taps.push(Tap { name, shape, data });
            }
            Ok(())
        };

        // front end: frames, DFT, log power, centered per frequency bin
        let samples = audio.len();
        let frames = 1 + (samples - 1) / HOP;
        let x = gpu.upload(audio.to_vec())?;
        let frame_buf = gpu.alloc(frames * N_FFT * 4)?;
        steps.frames(&frame_buf, &x, samples, &self.window)?;
        let spectrum = gpu.alloc(frames * 2 * BINS * 4)?;
        steps.dft(&spectrum, &frame_buf, &self.basis, frames)?;
        let log_power = gpu.alloc(BINS * frames * 4)?;
        steps.log_power(&log_power, &spectrum, frames)?;
        drop((frame_buf, spectrum));

        let max_grid = Grid::trunk(
            BINS,
            self.trunk[0].convs[0].co.max(self.trunk[19].convs[0].co),
            frames,
        );
        let bufs: Vec<Buf> = (0..4)
            .map(|_| gpu.alloc(max_grid.len() * 4))
            .collect::<Result<_>>()?;
        let [mut cur, mut next, t1, t2]: [Buf; 4] =
            bufs.try_into().map_err(|_| eyre!("four buffers"))?;
        let mut g = steps.center_rows(&cur, &log_power, BINS, frames)?;
        drop(log_power);
        tap("spec", &cur, g)?;

        let mean = gpu.alloc(HEAD * 4)?;
        let scale = gpu.alloc(HEAD * 4)?;
        for block in &self.trunk {
            let [a, b, c] = &block.convs;
            let og = steps.conv3x3(
                self.conv_math(g.channels == 1),
                (&cur, g),
                &t1,
                &a.w,
                a.co,
                block.stride,
            )?;
            steps.epilogue(
                &t1,
                0,
                og,
                slice(&a.bias, 0, a.co),
                slice(&a.bn, 0, 3 * a.co),
                true,
            )?;
            steps.conv3x3(self.conv_math(false), (&t1, og), &t2, &b.w, b.co, 1)?;
            steps.epilogue(
                &t2,
                0,
                og,
                slice(&b.bias, 0, b.co),
                slice(&b.bn, 0, 3 * b.co),
                true,
            )?;
            steps.conv3x3(self.conv_math(false), (&t2, og), &t1, &c.w, c.co, 1)?;
            steps.epilogue(
                &t1,
                0,
                og,
                slice(&c.bias, 0, c.co),
                slice(&c.bn, 0, 3 * c.co),
                true,
            )?;
            steps.channel_mean(&mean, &t1, og)?;
            steps.se_fc(&scale, &mean, &block.se)?;

            let residual = match (block.residual, &block.downsample) {
                (Residual::Identity, _) => Some(&cur),
                (Residual::Downsample(_), Some((w, bn))) => {
                    let dg =
                        steps.conv1x1_s2(self.conv_math(true), &cur, g, &t2, w, og.channels)?;
                    ensure!(dg == og, "{}: downsample {dg:?} against {og:?}", block.name);
                    steps.epilogue(&t2, 0, dg, Arg::EMPTY, slice(bn, 0, 3 * dg.channels), false)?;
                    Some(&t2)
                }
                (Residual::Downsample(_), None) => {
                    unreachable!("downsample blocks load their weights")
                }
                (Residual::None, _) => None,
            };
            if let Some(r) = residual {
                ensure!(
                    r.ptr() != cur.ptr() || g == og,
                    "{}: identity residual changes shape",
                    block.name
                );
            }
            steps.scale_residual(&next, &t1, &scale, residual, og)?;
            std::mem::swap(&mut cur, &mut next);
            g = og;
            tap(block.name, &cur, g)?;
        }

        // 1D head over frames; the trunk's interior rows and channels are one inner
        // dimension of stride `cols`
        ensure!(
            g.height() * g.channels == self.tdnn1.ci,
            "trunk ends at {g:?}"
        );
        let hg = Grid::head(HEAD, frames);
        let h1 = gpu.alloc(hg.len() * 4)?;
        steps.conv1d(
            self.conv_math(false),
            (at(&cur, g.origin()), g.cols),
            h1.ptr(),
            self.tdnn1.w.ptr(),
            (self.tdnn1.ci, HEAD, frames),
            0.0,
        )?;
        self.conv1_epilogue(&steps, &h1, &self.tdnn1, frames, true)?;
        drop((cur, next, t1, t2));
        tap("tdnn_1", &h1, hg)?;

        let r_in = gpu.alloc(hg.len() * 4)?;
        self.conv1(&steps, &h1, &r_in, &self.tdnn2_in, frames, true)?;
        let res = gpu.alloc(hg.len() * 4)?;
        self.res2net(&steps, &r_in, &res, frames)?;
        let r_out = gpu.alloc(hg.len() * 4)?;
        self.conv1(&steps, &res, &r_out, &self.tdnn2_out, frames, true)?;
        steps.channel_mean(&mean, &r_out, hg)?;
        steps.se_fc(&scale, &mean, &self.tdnn2_se)?;
        let h2 = r_in;
        steps.scale_residual(&h2, &r_out, &scale, Some(&h1), hg)?;
        drop((h1, res, r_out));
        tap("tdnn_2", &h2, hg)?;

        let pg = Grid::head(POOL_IN, frames);
        let d1 = gpu.alloc(pg.len() * 4)?;
        self.conv1(&steps, &h2, &d1, &self.dense1, frames, true)?;
        drop(h2);
        tap("dense_1", &d1, pg)?;

        let cat = gpu.alloc(3 * pg.len() * 4)?;
        steps.stats_cat(&cat, &d1, POOL_IN, frames)?;
        let att = gpu.alloc(ATTENTION * frames * 4)?;
        self.conv1(&steps, &cat, &att, &self.attention_in, frames, true)?;
        drop(cat);
        let logits = gpu.alloc(pg.len() * 4)?;
        self.conv1(&steps, &att, &logits, &self.attention_out, frames, false)?;
        let pooled = gpu.alloc(2 * POOL_IN * 4)?;
        steps.attentive_pool(&pooled, &logits, &d1, &self.pool_bn, POOL_IN, frames)?;
        if let Some(taps) = taps.as_deref_mut() {
            let pairs: Vec<f32> = gpu.download(&pooled)?;
            let data = (0..2)
                .flat_map(|part| (0..POOL_IN).map(move |c| (c, part)))
                .map(|(c, part)| pairs[2 * c + part])
                .collect();
            taps.push(Tap {
                name: "pool",
                shape: vec![2 * POOL_IN],
                data,
            });
        }

        let emb = gpu.alloc(DIM * 4)?;
        gpu.copy_dtod(self.dense2_b.ptr(), emb.ptr(), DIM * 4)?;
        blas.gemm_f32(
            F32Math::Full,
            (true, false),
            (DIM, 1, 2 * POOL_IN),
            Operand::new(self.dense2_w.ptr(), 2 * POOL_IN),
            Operand::new(pooled.ptr(), 2 * POOL_IN),
            Operand::new(emb.ptr(), DIM),
            1,
            1.0,
        )?;
        let out: Vec<f32> = gpu.download(&emb)?;
        if let Some(taps) = taps {
            taps.push(Tap {
                name: "embedding",
                shape: vec![DIM],
                data: out.clone(),
            });
        }
        ensure!(
            out.iter().all(|v| v.is_finite()),
            "non-finite ECAPA2 embedding"
        );
        Ok(out)
    }

    /// 1x1 conv1d of `x` into `y`, then bias, ReLU when `relu`, and the conv's batch norm
    fn conv1(
        &self,
        steps: &Steps,
        x: &Buf,
        y: &Buf,
        conv: &Conv1,
        frames: usize,
        relu: bool,
    ) -> Result<()> {
        steps.conv1d(
            self.conv_math(false),
            (x.ptr(), frames),
            y.ptr(),
            conv.w.ptr(),
            (conv.ci, conv.co, frames),
            0.0,
        )?;
        self.conv1_epilogue(steps, y, conv, frames, relu)
    }

    fn conv1_epilogue(
        &self,
        steps: &Steps,
        y: &Buf,
        conv: &Conv1,
        frames: usize,
        relu: bool,
    ) -> Result<()> {
        let bn = conv
            .bn
            .as_ref()
            .map_or(Arg::EMPTY, |b| slice(b, 0, 3 * conv.co));
        steps.epilogue(
            y,
            0,
            Grid::head(conv.co, frames),
            slice(&conv.bias, 0, conv.co),
            bn,
            relu,
        )
    }

    /// Res2Net: group 0 passes through; group i >= 1 is
    /// `bn(relu(conv(reflect_pad(x_i + y_{i-1}))))` (no `y_0` term for i = 1), with the
    /// dilated 3-tap conv as three accumulating GEMMs over shifted views
    fn res2net(&self, steps: &Steps, x: &Buf, out: &Buf, frames: usize) -> Result<()> {
        let group = RES2_WIDTH * frames;
        let pad = RES2_DILATION;
        let padded = steps.gpu.alloc(RES2_WIDTH * (frames + 2 * pad) * 4)?;
        steps.gpu.copy_dtod(x.ptr(), out.ptr(), group * 4)?;
        for (i, conv) in self.res2.iter().enumerate().map(|(i, c)| (i + 1, c)) {
            let prev = if i == 1 {
                Arg::EMPTY
            } else {
                slice(out, (i - 1) * group, group)
            };
            steps.reflect_add(
                &padded,
                slice(x, i * group, group),
                prev,
                (RES2_WIDTH, frames, pad),
            )?;
            for k in 0..3 {
                steps.conv1d(
                    self.conv_math(false),
                    (at(&padded, k * RES2_DILATION), frames + 2 * pad),
                    at(out, i * group),
                    at(&conv.w, k * RES2_WIDTH * RES2_WIDTH),
                    (RES2_WIDTH, RES2_WIDTH, frames),
                    if k == 0 { 0.0 } else { 1.0 },
                )?;
            }
            steps.epilogue(
                out,
                i * group,
                Grid::head(RES2_WIDTH, frames),
                slice(&conv.bias, 0, RES2_WIDTH),
                slice(&conv.bn, 0, 3 * RES2_WIDTH),
                true,
            )?;
        }

        Ok(())
    }
}

/// `F.normalize(e, dim=-1)`: `e / max(|e|, 1e-12)`, as `_embed_audio_segment` returns it
pub fn normalize(e: &[f32]) -> Vec<f32> {
    let norm = e
        .iter()
        .map(|&v| v as f64 * v as f64)
        .sum::<f64>()
        .sqrt()
        .max(1e-12);
    e.iter().map(|&v| (v as f64 / norm) as f32).collect()
}

/// torch `[co][ci][kh][kw]` conv weights to `[kw][co][kh * ci]`: per kernel column, a
/// matrix whose inner index walks kernel rows then input channels, the order the trunk
/// layout gives a shifted view
pub fn conv3x3_layout(w: &[f32], co: usize, ci: usize) -> Vec<f32> {
    let mut out = vec![0f32; w.len()];
    for o in 0..co {
        for c in 0..ci {
            for kh in 0..3 {
                for kw in 0..3 {
                    out[(kw * co + o) * 3 * ci + kh * ci + c] = w[((o * ci + c) * 3 + kh) * 3 + kw];
                }
            }
        }
    }

    out
}

/// The `[N_FFT][2 BINS]` real DFT basis: `cos(2 pi k n / N)` in the first `BINS` columns
/// and `-sin(2 pi k n / N)` in the rest, evaluated in f64 with the angle reduced mod N
pub fn dft_basis() -> Vec<f32> {
    let mut out = vec![0f32; N_FFT * 2 * BINS];
    for n in 0..N_FFT {
        for k in 0..BINS {
            let angle = std::f64::consts::TAU * ((k * n) % N_FFT) as f64 / N_FFT as f64;
            out[n * 2 * BINS + k] = angle.cos() as f32;
            out[n * 2 * BINS + BINS + k] = -angle.sin() as f32;
        }
    }

    out
}
