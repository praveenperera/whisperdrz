//! GPU runtime: the CUDA device, torch-identical cuBLAS GEMMs and the cuda-oxide kernels

pub mod blas;
pub mod device;

pub use blas::Blas;
pub use device::{Arg, Buf, Gpu, blocks};
