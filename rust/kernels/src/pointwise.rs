//! FP16 pointwise kernels of the transformer blocks and the conv stem, eight halves per
//! thread with 16-byte loads and stores
//!
//! Each op computes in FP32 and rounds once to FP16 where torch's kernel does, so a fused
//! kernel keeps every intermediate rounding of the separate torch ops it replaces

use core::f32::consts::FRAC_1_SQRT_2;

use cuda_device::convert::cvt_f32x2_f16x2;
use cuda_device::{DisjointSlice, kernel, launch_contract, thread};

use crate::common::{f2h, h2f};
use crate::mma_f16::pack_f16;

/// Eight FP16 values in one 16-byte word, so they move in one vector access
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct H8(pub [u32; 4]);

impl H8 {
    /// The eight values widened to FP32 (exact)
    #[inline(always)]
    pub fn unpack(self) -> [f32; 8] {
        let (a, b) = cvt_f32x2_f16x2(self.0[0]);
        let (c, d) = cvt_f32x2_f16x2(self.0[1]);
        let (e, f) = cvt_f32x2_f16x2(self.0[2]);
        let (g, h) = cvt_f32x2_f16x2(self.0[3]);
        [a, b, c, d, e, f, g, h]
    }

    /// Eight floats rounded to nearest-even FP16
    #[inline(always)]
    pub fn pack(v: [f32; 8]) -> Self {
        Self([
            pack_f16(v[0], v[1]),
            pack_f16(v[2], v[3]),
            pack_f16(v[4], v[5]),
            pack_f16(v[6], v[7]),
        ])
    }
}

/// Exact (erf) GELU in FP32 as torch's CUDA kernel computes it for FP16 tensors:
/// `x * 0.5 * (1 + erf(x * sqrt(1/2)))`
#[inline(always)]
pub fn gelu(x: f32) -> f32 {
    x * 0.5 * (1.0 + libm::erff(x * FRAC_1_SQRT_2))
}

/// GELU on FP16 in place, `x = f16(gelu(x))`, eight values per thread; `x` is counted in
/// groups of eight
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn gelu_f16x8(mut x: DisjointSlice<H8>) {
    let t = thread::index_1d_u32(lc);
    let Some(mut x) = x.element_thread32(t) else {
        return;
    };
    let mut v = x.read().unpack();
    #[unroll]
    for e in 0..8 {
        v[e] = gelu(v[e]);
    }
    x.write(H8::pack(v));
}

/// Residual add on FP16 in place, `x = f16(x + d)`, eight values per thread
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn add_f16x8(mut x: DisjointSlice<H8>, d: &[H8]) {
    let t = thread::index_1d_u32(lc);
    let i = t.get() as usize;
    let Some(mut x) = x.element_thread32(t) else {
        return;
    };
    let (mut a, b) = (x.read().unpack(), d[i].unpack());
    #[unroll]
    for e in 0..8 {
        a[e] += b[e];
    }
    x.write(H8::pack(a));
}

/// The end of the encoder stem: `out = f16(f16(gelu(y)) + pos[i % pos.len()])`, the
/// second conv's GELU and the positional embedding broadcast over the batch
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn gelu_pos_f16x8(mut out: DisjointSlice<H8>, y: &[H8], pos: &[H8]) {
    let t = thread::index_1d_u32(lc);
    let i = t.get() as usize;
    let Some(mut out) = out.element_thread32(t) else {
        return;
    };
    let (mut a, p) = (y[i].unpack(), pos[i % pos.len()].unpack());
    #[unroll]
    for e in 0..8 {
        // torch rounds the GELU to FP16 before the add
        a[e] = h2f(f2h(gelu(a[e]))) + p[e];
    }
    out.write(H8::pack(a));
}

/// im2col for a kernel-3, padding-1 Conv1d, so the conv runs as one GEMM against its
/// weight viewed as `[c_out, c_in * 3]`
///
/// `out` is `[batch * t_out, c_in * 3]` with column `c * 3 + j` (the weight's own memory
/// order) holding input channel `c` at time `t * stride + j - 1`, zero outside `[0,
/// t_in)`. Input element `(b, c, t)` is `x[b * x_bs + c * x_cs + t * x_ts]`, which covers
/// both the channel-major mel and the time-major output of a previous GEMM. With `gelu`
/// nonzero each input value goes through `f16(gelu(.))` first, fusing the previous conv's
/// activation into the gather (the zero padding stays zero, as torch pads after it)
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn im2col_k3_f16(
    mut out: DisjointSlice<u16>,
    x: &[u16],
    t_in: u32,
    t_out: u32,
    c_in: u32,
    stride: u32,
    x_bs: u32,
    x_cs: u32,
    x_ts: u32,
    gelu_in: u32,
) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let Some(mut out) = out.element_thread32(t) else {
        return;
    };
    let width = c_in * 3;
    let (row, col) = (i / width, i % width);
    let (b, tt) = (row / t_out, row % t_out);
    let (c, j) = (col / 3, col % 3);
    // time tt * stride + j - 1, kept unsigned: 0 stands for the left padding
    let ti = tt * stride + j;
    let mut v = 0u16;
    if ti >= 1 && ti <= t_in {
        v = x[b as usize * x_bs as usize
            + c as usize * x_cs as usize
            + (ti - 1) as usize * x_ts as usize];
        if gelu_in != 0 {
            v = f2h(gelu(h2f(v)));
        }
    }
    out.write(v);
}
