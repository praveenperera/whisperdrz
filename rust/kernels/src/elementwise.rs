//! Elementwise kernels: one thread per output element, 32-bit index math

use cuda_device::{DisjointSlice, kernel, launch_contract, thread};

use crate::common::{f2h, h2f};

/// Exact (erf) GELU on FP16, computed in FP32 as PyTorch's CUDA kernel does:
/// `x * 0.5 * (1 + erf(x * sqrt(1/2)))`, rounded once to FP16
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn gelu_f16(mut out: DisjointSlice<u16>, x: &[u16]) {
    let t = thread::index_1d_u32(lc);
    let i = t.get() as usize;
    let Some(mut out) = out.element_thread32(t) else { return };
    let v = h2f(x[i]);
    out.write(f2h(v * 0.5 * (1.0 + libm::erff(v * core::f32::consts::FRAC_1_SQRT_2))));
}
