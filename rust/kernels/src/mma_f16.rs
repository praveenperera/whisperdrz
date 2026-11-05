//! FP16 tensor-core operands: shared-memory tiles, `ldmatrix` fragments, `cp.async` copies
//! and the m16n8k16 MMA with FP32 accumulation
//!
//! The types keep a generic or global pointer out of the slots that take a 32-bit
//! shared-window address ([`SharedAddr`]) and an A fragment out of a B slot. Fragments only
//! come from a tile load or from the accumulator-to-A identity. Every type is a transparent
//! wrapper over registers or an address, so none costs an instruction. Adapted from the
//! sam3rs kernels, where the same layouts run the BF16 and FP16 flash attention

use cuda_device::convert::cvt_f16x2_f32;
use cuda_device::ptx_asm;
use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::thread::threadIdx_x;
use cuda_device::wmma::{
    ldmatrix_x4_shared_u32, ldmatrix_x4_trans_shared_u32, mma_m16n8k16_f32_f16,
};

/// A byte address in the CTA's shared-memory window, the operand `ldmatrix` and
/// `cp.async` take; only a [`SharedTile`] makes one
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct SharedAddr(u32);

/// Two floats rounded to nearest-even FP16, `lo` in the low half
#[inline(always)]
pub fn pack_f16(lo: f32, hi: f32) -> u32 {
    cvt_f16x2_f32(lo, hi)
}

/// Starts a 16-byte copy from global `src` to shared `dst` that bypasses L1
/// (`cp.async.cg`); with `full` false it reads nothing and writes 16 zero bytes
///
/// # Safety
/// `dst` is 16-byte aligned inside its tile, and `src` is 16-byte aligned with 16 readable
/// bytes when `full`. No thread touches `dst` until this thread has waited on the copy
/// ([`cp_async_wait_all`]) and a barrier has followed
#[inline(always)]
pub unsafe fn cp_async16(dst: SharedAddr, src: *const u16, full: bool) {
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %1; cp.async.cg.shared.global [%0], [g], 16, %2; }",
            in("r") dst.0,
            in("l") src as u64,
            in("r") if full { 16u32 } else { 0u32 },
            clobber("memory"),
        );
    }
}

/// Closes the group of this thread's copies started since the last commit
///
/// # Safety
/// None beyond [`cp_async16`]'s, but kept unsafe to sit with the copies it orders
#[inline(always)]
pub unsafe fn cp_async_commit() {
    unsafe { ptx_asm!("cp.async.commit_group;", clobber("memory")) };
}

/// Waits until every copy this thread started has landed; other threads' copies need a
/// barrier after it
///
/// # Safety
/// As [`cp_async_commit`]
#[inline(always)]
pub unsafe fn cp_async_wait_all() {
    unsafe { ptx_asm!("cp.async.wait_group 0;", clobber("memory")) };
}

/// `c += a @ b` on FP16 operands with FP32 accumulation, as FlashAttention-2 and cuBLAS
/// run FP16 inputs
///
/// # Safety
/// All 32 lanes of the warp call it together
#[inline(always)]
pub unsafe fn mma(c: [f32; 4], a: FragA, b: FragB) -> [f32; 4] {
    unsafe { mma_m16n8k16_f32_f16(c, a.0, b.0) }
}

/// One lane's registers of a row-major 16x16 FP16 A operand
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct FragA([u32; 4]);

impl FragA {
    pub const ZERO: Self = Self([0; 4]);

    /// The fragment from its four registers: rows `lane / 4` and the same plus 8 at depth
    /// `2 * (lane % 4)`, then both at that depth plus 8, low half first. For an A operand
    /// read straight from global memory, as the small-M linears keep their weights in
    /// registers
    #[inline(always)]
    pub fn from_bits(a: [u32; 4]) -> Self {
        Self(a)
    }

    /// The A operand whose left and right 16x8 halves are the FP32 accumulators `lo` and
    /// `hi`, each value rounded to FP16: an m16n8 accumulator has the register layout of
    /// half an A fragment, which is how P = softmax(S) feeds P @ V without a trip through
    /// shared memory
    #[inline(always)]
    pub fn from_acc_pair(lo: [f32; 4], hi: [f32; 4]) -> Self {
        Self([
            pack_f16(lo[0], lo[1]),
            pack_f16(lo[2], lo[3]),
            pack_f16(hi[0], hi[1]),
            pack_f16(hi[2], hi[3]),
        ])
    }
}

/// One lane's registers of a column-major 16x8 FP16 B operand
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct FragB([u32; 2]);

impl FragB {
    pub const ZERO: Self = Self([0; 2]);

    /// The fragment from its two registers: `b[0]` holds B rows `2 * (lane % 4)` and the
    /// next, `b[1]` the same rows plus 8, both in column `lane / 4`, low half first. For a
    /// B operand read straight from global memory, as the absorbed cross-attention keeps its
    /// queries in registers for the whole kernel
    #[inline(always)]
    pub fn from_bits(b: [u32; 2]) -> Self {
        Self(b)
    }
}

/// A row-major tile of FP16 in shared memory with rows of `LD` elements
#[derive(Clone, Copy)]
pub struct SharedTile<const LD: usize> {
    base: SharedAddr,
}

impl<const LD: usize> SharedTile<LD> {
    /// # Safety
    /// `base` points to a 16-byte aligned shared-memory array of FP16 values, as many rows
    /// of `LD` as callers address
    #[inline(always)]
    pub unsafe fn new(base: *mut u16) -> Self {
        Self {
            base: SharedAddr(unsafe { cvta_generic_to_shared_u32(base as *const u8) }),
        }
    }

    /// The address of element `(row, col)`
    #[inline(always)]
    pub fn at(self, row: usize, col: usize) -> SharedAddr {
        // 32-bit math: an offset computed in 64 bits and truncated hides its constant part
        // from ptxas, which then can't fold it into the access's immediate
        SharedAddr(self.base.0 + (row as u32 * LD as u32 + col as u32) * 2)
    }

    /// The address of `(row, col)` plus this lane's `ldmatrix.x4` row: matrix `mi`'s row
    /// `ri`, at `(mi % 2 * 8 + ri, mi / 2 * 8)` when `ROW_HALF_FIRST`, else at `(mi / 2 * 8
    /// + ri, mi % 2 * 8)`
    #[inline(always)]
    fn ldsm_addr<const ROW_HALF_FIRST: bool>(self, row: usize, col: usize) -> SharedAddr {
        let lane = threadIdx_x() % 32;
        let (mi, ri) = (lane / 8, lane % 8);
        let (lr, lc) = if ROW_HALF_FIRST {
            ((mi & 1) * 8 + ri, (mi / 2) * 8)
        } else {
            ((mi / 2) * 8 + ri, (mi & 1) * 8)
        };
        // the lane's part first and the fragment's constant offset last: summed the other
        // way, LLVM merges the two into an `or` per load, which ptxas can't fold into the
        // load's immediate
        let lane_addr = self.base.0 + (lr * LD as u32 + lc) * 2;
        SharedAddr(lane_addr + (row as u32 * LD as u32 + col as u32) * 2)
    }

    /// The A fragment of rows `[m0, m0 + 16)` and columns `[k0, k0 + 16)`
    ///
    /// # Safety
    /// All 32 lanes call it together with the same `m0` and `k0`, inside the tile, after
    /// a barrier that orders the tile's stores
    #[inline(always)]
    pub unsafe fn frag_a(self, m0: usize, k0: usize) -> FragA {
        // matrices: rows m0.. at k0, m0 + 8.. at k0, m0.. at k0 + 8, m0 + 8.. at k0 + 8,
        // the A register order
        let p = self.ldsm_addr::<true>(m0, k0);
        FragA(unsafe { ldmatrix_x4_shared_u32(p.0) })
    }

    /// The A fragment of rows `[m0, m0 + 16)` and columns `[k0, k0 + 16)` from a tile that
    /// holds A transposed: tile rows are A's columns, at `(k0, m0)`
    ///
    /// # Safety
    /// As [`SharedTile::frag_a`]
    #[inline(always)]
    pub unsafe fn frag_a_trans(self, k0: usize, m0: usize) -> FragA {
        // transposed matrices in the A register order: depth k0.. at m0, k0.. at m0 + 8,
        // k0 + 8.. at m0, k0 + 8.. at m0 + 8
        let p = self.ldsm_addr::<false>(k0, m0);
        FragA(unsafe { ldmatrix_x4_trans_shared_u32(p.0) })
    }

    /// The B fragments of columns `[n0, n0 + 8)` and `[n0 + 8, n0 + 16)`, depth `[k0, k0 +
    /// 16)`, from a tile that holds B transposed: rows are B's columns, as K holds keys
    ///
    /// # Safety
    /// As [`SharedTile::frag_a`]
    #[inline(always)]
    pub unsafe fn frag_b_pair_nk(self, n0: usize, k0: usize) -> [FragB; 2] {
        // matrices: n0.. at k0 and k0 + 8, then n0 + 8.. at k0 and k0 + 8
        let p = self.ldsm_addr::<false>(n0, k0);
        let r = unsafe { ldmatrix_x4_shared_u32(p.0) };
        [FragB([r[0], r[1]]), FragB([r[2], r[3]])]
    }

    /// The same from a tile that holds B as is, rows along the depth, as V does
    ///
    /// # Safety
    /// As [`SharedTile::frag_a`]
    #[inline(always)]
    pub unsafe fn frag_b_pair_kn(self, k0: usize, n0: usize) -> [FragB; 2] {
        // transposed matrices: depth k0.. and k0 + 8.. at n0, then the same at n0 + 8
        let p = self.ldsm_addr::<true>(k0, n0);
        let r = unsafe { ldmatrix_x4_trans_shared_u32(p.0) };
        [FragB([r[0], r[1]]), FragB([r[2], r[3]])]
    }
}
