//! cuda-oxide device kernels for whisperdrz
//!
//! `scripts/build-ptx.sh` compiles this crate for compute_80 and writes
//! `kernels/ptx/whisperdrz.sm80.ptx`, which the whisperdrz-native crate embeds and loads by
//! kernel name through the driver. Buffers are `&[T]` inputs and `DisjointSlice<T>`
//! outputs, each two PTX params (address, then length in elements); other sizes are `u32`
//! and other scalars `f32`. Half-precision values travel as their `u16` bit patterns,
//! since the `f16` primitive is unstable on the pinned stable toolchain

#![no_std]

pub mod attention;
pub mod common;
pub mod cross;
pub mod decoder;
pub mod dtw;
pub mod ecapa2;
pub mod elementwise;
pub mod flash;
pub mod linear;
pub mod mma_f16;
pub mod norm;
pub mod pointwise;
