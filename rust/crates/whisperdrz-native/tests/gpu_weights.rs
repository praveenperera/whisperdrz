//! The checkpoint round-trips through the GPU loader bit for bit
//!
//! Skips unless `WHISPERDRZ_WEIGHTS` names `whisperdrz-large-v3.safetensors`; run through
//! the GPU queue with `rust/scripts/gpu-test.sh`
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use eyre::{Result, ensure};
use safetensors::SafeTensors;
use whisperdrz_native::cuda::{Gpu, Weights};

#[test]
fn checkpoint_round_trips() -> Result<()> {
    let Some(path) = std::env::var_os("WHISPERDRZ_WEIGHTS").map(PathBuf::from) else {
        eprintln!("skipping: WHISPERDRZ_WEIGHTS not set (run scripts/gpu-test.sh)");
        return Ok(());
    };
    let gpu = Gpu::new(0)?;
    // the transcriber never runs the speaker head
    let weights = Weights::load(&gpu, &path, |name| !name.starts_with("speaker_head."))?;
    ensure!(weights.len() == 1259, "{} tensors loaded", weights.len());

    let bytes = std::fs::read(&path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    for (name, shape) in [
        ("encoder.conv1.weight", vec![1280, 128, 3]),
        ("decoder.token_embedding.weight", vec![51866, 1280]),
        ("decoder.blocks.31.mlp.2.bias", vec![1280]),
    ] {
        let t = weights.get(name, &shape)?;
        let got: Vec<u8> = gpu.download(&t.buf)?;
        ensure!(
            got == st.tensor(name)?.data(),
            "{name} differs after the round trip"
        );
    }
    println!(
        "loaded {} tensors, {} bytes",
        weights.len(),
        weights.bytes()
    );
    Ok(())
}
