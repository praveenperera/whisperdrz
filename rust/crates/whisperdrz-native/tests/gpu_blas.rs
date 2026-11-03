//! Native cuBLAS calls against torch's FP16 GEMMs, bit for bit
//!
//! `rust/tools/gemm_ref.py` writes the torch side; `rust/scripts/gpu-test.sh` runs both
//! through the GPU queue. Skips when `WHISPERDRZ_GEMM_REF` doesn't name its output. The
//! gate doesn't need bitwise GEMMs, but with them a per-layer diff against PyTorch points
//! at our own kernels rather than at cuBLAS
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use eyre::{Result, bail, ensure};
use serde_json::Value;
use whisperdrz_fixtures::{ArrayData, read_npy};
use whisperdrz_native::cuda::{Blas, Gpu};

fn f16_bits(path: &Path) -> Result<(Vec<usize>, Vec<u16>)> {
    let a = read_npy(path)?;
    let ArrayData::F16(v) = a.data else {
        bail!("{} is not float16", path.display())
    };
    Ok((a.shape, v.iter().map(|h| h.to_bits()).collect()))
}

#[test]
fn gemms_match_torch_bitwise() -> Result<()> {
    let Some(dir) = std::env::var_os("WHISPERDRZ_GEMM_REF").map(PathBuf::from) else {
        eprintln!("skipping: WHISPERDRZ_GEMM_REF not set (run scripts/gpu-test.sh)");
        return Ok(());
    };
    let manifest: Value = serde_json::from_slice(&std::fs::read(dir.join("cases.json"))?)?;
    let cases = manifest["cases"].as_array().cloned().unwrap_or_default();
    ensure!(!cases.is_empty(), "no GEMM cases in {}", dir.display());

    let gpu = Gpu::new(0)?;
    let blas = Blas::new(&gpu)?;
    let mut failed = Vec::new();
    for case in &cases {
        let field = |k: &str| case[k].as_str().map(|f| dir.join(f));
        let dim = |k: &str| case[k].as_u64().unwrap_or(0) as usize;
        let (kind, rows, n_in, n_out) = (
            case["kind"].as_str().unwrap_or(""),
            dim("rows"),
            dim("n_in"),
            dim("n_out"),
        );

        let (_, x) = f16_bits(&field("x").unwrap())?;
        let (_, w) = f16_bits(&field("w").unwrap())?;
        let (y_shape, want) = f16_bits(&field("y").unwrap())?;
        ensure!(
            y_shape == [rows, n_out],
            "{kind} {rows}: torch output shape {y_shape:?}"
        );
        ensure!(
            x.len() == rows * n_in && w.len() == n_in * n_out,
            "{kind} {rows}: operand sizes"
        );

        let x = gpu.upload(x)?;
        let w = gpu.upload(w)?;
        // NaN-filled, so a GEMM that writes nothing can't pass
        let y = gpu.upload(vec![0x7e00u16; rows * n_out])?;
        match kind {
            "bias" => {
                let (_, b) = f16_bits(&field("b").unwrap())?;
                let b = gpu.upload(b)?;
                blas.linear_bias(x.ptr(), w.ptr(), b.ptr(), y.ptr(), (rows, n_in, n_out))?;
                gpu.synchronize()?;
            }
            "nobias" => blas.linear(x.ptr(), w.ptr(), y.ptr(), (rows, n_in, n_out))?,
            "lmhead" => blas.matmul(x.ptr(), w.ptr(), y.ptr(), (rows, n_in, n_out))?,
            other => bail!("unknown GEMM case kind {other}"),
        }
        let got: Vec<u16> = gpu.download(&y)?;

        ensure!(got.len() == want.len(), "{kind} {rows}: output length");
        let diff = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        let max_abs = got
            .iter()
            .zip(&want)
            .map(|(a, b)| {
                (half::f16::from_bits(*a).to_f32() - half::f16::from_bits(*b).to_f32()).abs()
            })
            .fold(0f32, |m, d| if d.is_nan() || d > m { d } else { m });
        let label = format!("{kind} rows={rows} {n_in}->{n_out}");
        println!("{label}: {diff}/{} differ, max |diff| {max_abs}", got.len());
        if diff > 0 {
            failed.push(label);
        }
    }

    ensure!(
        failed.is_empty(),
        "{} of {} GEMMs not bitwise equal to torch: {failed:?}",
        failed.len(),
        cases.len()
    );
    Ok(())
}
