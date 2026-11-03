//! The whisperdrz checkpoint on the GPU
//!
//! `whisperdrz-large-v3.safetensors` holds every tensor as FP16: the encoder and decoder
//! blocks, conv stem, positional and token embeddings, and a `speaker_head` the
//! transcriber never runs. Tensors upload one by one from the memory-mapped file, so
//! loading never holds a second host copy of the checkpoint

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use eyre::{Result, WrapErr, ensure, eyre};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use super::{Buf, Gpu};

/// An FP16 tensor in device memory, row-major
pub struct Tensor {
    pub buf: Buf,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn ptr(&self) -> u64 {
        self.buf.ptr()
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Checkpoint tensors by name
pub struct Weights(HashMap<String, Tensor>);

impl Weights {
    /// Uploads every tensor whose name `keep` accepts
    pub fn load(gpu: &Gpu, path: &Path, keep: impl Fn(&str) -> bool) -> Result<Self> {
        let file = File::open(path).wrap_err_with(|| format!("open {}", path.display()))?;
        // the file is read-only and nothing else in the process writes it
        let map =
            unsafe { Mmap::map(&file) }.wrap_err_with(|| format!("map {}", path.display()))?;
        let st = SafeTensors::deserialize(&map)
            .wrap_err_with(|| format!("read safetensors header of {}", path.display()))?;

        let mut tensors = HashMap::new();
        for (name, view) in st.tensors() {
            if !keep(&name) {
                continue;
            }

            ensure!(
                view.dtype() == Dtype::F16,
                "{name} is {:?}; the checkpoint should be FP16 throughout",
                view.dtype()
            );
            let buf = gpu.upload(view.data().to_vec())?;
            let shape = view.shape().to_vec();
            tensors.insert(name, Tensor { buf, shape });
        }
        gpu.synchronize()?;

        Ok(Self(tensors))
    }

    /// The tensor `name`, checked against the shape the caller expects
    pub fn get(&self, name: &str, shape: &[usize]) -> Result<&Tensor> {
        let t = self
            .0
            .get(name)
            .ok_or_else(|| eyre!("checkpoint has no tensor {name}"))?;
        ensure!(
            t.shape == shape,
            "{name} has shape {:?}, want {shape:?}",
            t.shape
        );
        Ok(t)
    }

    /// Removes and returns the tensor `name`, checked against the shape the caller
    /// expects, for a consumer that repacks it and frees the original
    pub fn take(&mut self, name: &str, shape: &[usize]) -> Result<Tensor> {
        self.get(name, shape)?;
        Ok(self.0.remove(name).expect("checked by get"))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Device bytes held
    pub fn bytes(&self) -> usize {
        self.0.values().map(|t| t.buf.bytes()).sum()
    }
}
