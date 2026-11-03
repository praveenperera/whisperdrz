//! Helpers shared by the kernel modules

use cuda_device::convert;

/// Widens the bits of one IEEE half to `f32` (exact)
#[inline(always)]
pub fn h2f(bits: u16) -> f32 {
    convert::cvt_f32_f16x2_lo(bits as u32)
}

/// Rounds `f32` to the nearest IEEE half (ties to even) and returns its bits
#[inline(always)]
pub fn f2h(value: f32) -> u16 {
    convert::cvt_f16x2_f32(value, 0.0) as u16
}
