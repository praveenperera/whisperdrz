//! The native encoder against torch, per layer and end to end
//!
//! Both sides are judged against a common truth, torch in FP32 on the same FP16 weights
//! and inputs (`rust/tools/encoder_layers.py`), never only against each other. A layer
//! passes when our max-abs and mean-abs errors against the truth are at most 1.5x torch
//! FP16's. Layers are checked in isolation (our layer N on torch FP16's layer N - 1
//! output, against the FP32 layer N on that same input) and chained from the mel (against
//! the FP32 chain), and the whole encoder on recorded batches from the reference traces.
//! The chained check also runs two other torch FP16 variants that each make one change
//! exact in real arithmetic (memory-efficient SDPA; convs as im2col + GEMM, the native
//! stem's lowering). Chained max-abs is judged against the worst of the three torch FP16
//! runs rather than the standard one alone: a single outlier element compounds through
//! the stack, and torch's own im2col variant lands 1.58x the standard run's max-abs at
//! block06, so the standard run alone is a sharper bar than FP16 itself justifies.
//! Chained mean-abs and every isolated and end-to-end check keep the standard run.
//! Skips unless `WHISPERDRZ_WEIGHTS` and `WHISPERDRZ_ENCODER_REF` are set; run through the
//! GPU queue with `rust/scripts/gpu-test.sh`
#![cfg(feature = "cuda")]

#[allow(dead_code)]
#[path = "support/numerics.rs"]
mod numerics;

use std::path::{Path, PathBuf};

use eyre::{Result, bail, ensure, eyre};
use numerics::{Err, NAN16};
use serde_json::Value;
use whisperdrz_fixtures::{ArrayData, TraceSet, read_npy};
use whisperdrz_native::cuda::encoder::{D_MODEL, N_CTX, Workspace};
use whisperdrz_native::cuda::{Blas, Buf, Encoder, Gpu, Weights};

/// Ours may be this much worse than torch FP16 against the FP32 truth
const MAX_RATIO: f64 = 1.5;

struct Env {
    refs: PathBuf,
    manifest: Value,
    gpu: Gpu,
    blas: Blas,
    encoder: Encoder,
}

fn env() -> Result<Option<Env>> {
    let (Some(weights), Some(refs)) = (
        std::env::var_os("WHISPERDRZ_WEIGHTS").map(PathBuf::from),
        std::env::var_os("WHISPERDRZ_ENCODER_REF").map(PathBuf::from),
    ) else {
        eprintln!(
            "skipping: WHISPERDRZ_WEIGHTS or WHISPERDRZ_ENCODER_REF not set (run scripts/gpu-test.sh)"
        );
        return Ok(None);
    };
    let manifest: Value = serde_json::from_slice(&std::fs::read(refs.join("manifest.json"))?)?;
    let gpu = Gpu::new(0)?;
    let blas = Blas::new(&gpu)?;
    let mut w = Weights::load(&gpu, &weights, |n| n.starts_with("encoder."))?;
    let encoder = Encoder::new(&gpu, &mut w)?;
    ensure!(w.is_empty(), "{} encoder tensors left unused", w.len());
    Ok(Some(Env {
        refs,
        manifest,
        gpu,
        blas,
        encoder,
    }))
}

/// An FP16 `.npy` as bits, with its shape
fn f16_bits(path: &Path) -> Result<(Vec<usize>, Vec<u16>)> {
    let a = read_npy(path)?;
    let ArrayData::F16(v) = a.data else {
        bail!("{} is not float16", path.display())
    };
    Ok((a.shape, v.iter().map(|h| h.to_bits()).collect()))
}

/// Any float `.npy` widened to f64
fn f64s(path: &Path) -> Result<Vec<f64>> {
    Ok(read_npy(path)?
        .to_f32()?
        .into_iter()
        .map(f64::from)
        .collect())
}

fn bits_f64(bits: &[u16]) -> Vec<f64> {
    bits.iter().copied().map(numerics::f16_to_f64).collect()
}

fn download(gpu: &Gpu, buf: &Buf, n: usize) -> Result<Vec<f64>> {
    let mut host = vec![0u16; n];
    gpu.download_into(buf, 0, &mut host)?;
    Ok(bits_f64(&host))
}

/// One comparison row: ours and torch FP16 against the truth
struct Row {
    name: String,
    ours: Err,
    torch: Err,
    /// The max-abs error ours is judged against: torch FP16's, or for chained layers the
    /// worst valid torch FP16 variant's
    max_bar: f64,
}

impl Row {
    fn new(name: &str, ours: &[f64], torch: &[f64], truth: &[f64]) -> Result<Self> {
        let torch = Err::of(torch, truth)?;
        Ok(Self {
            name: name.to_owned(),
            ours: Err::of(ours, truth)?,
            max_bar: torch.max_abs,
            torch,
        })
    }

    /// Raises the max-abs bar to the worst of these variant rows' own errors
    fn bar_from(mut self, variants: &[&Row]) -> Self {
        for v in variants {
            self.max_bar = self.max_bar.max(v.ours.max_abs);
        }
        self
    }

    fn passes(&self) -> bool {
        self.ours.max_abs <= MAX_RATIO * self.max_bar
            && self.ours.mean_abs <= MAX_RATIO * self.torch.mean_abs
    }
}

/// Prints a table and returns the names of the rows that fail; a table that is not
/// `checked` prints no failure marks and returns none
fn report(title: &str, rows: &[Row], checked: bool) -> Vec<String> {
    println!("\n{title}");
    println!(
        "{:<8} {:>11} {:>11} {:>11} {:>6} {:>11} {:>11} {:>6}",
        "layer", "ours max", "torch max", "max bar", "ratio", "ours mean", "torch mean", "ratio"
    );
    for r in rows {
        println!(
            "{:<8} {:>11.4e} {:>11.4e} {:>11.4e} {:>6.2} {:>11.4e} {:>11.4e} {:>6.2}{}",
            r.name,
            r.ours.max_abs,
            r.torch.max_abs,
            r.max_bar,
            r.ours.max_abs / r.max_bar,
            r.ours.mean_abs,
            r.torch.mean_abs,
            r.ours.mean_abs / r.torch.mean_abs,
            if !checked || r.passes() { "" } else { "  FAIL" }
        );
    }
    rows.iter()
        .filter(|r| checked && !r.passes())
        .map(|r| format!("{title}: {}", r.name))
        .collect()
}

#[test]
fn encoder_layers_match_torch() -> Result<()> {
    let Some(mut env) = env()? else { return Ok(()) };
    let layers = env.manifest["layers"].clone();
    let names: Vec<String> = layers["names"]
        .as_array()
        .ok_or_else(|| eyre!("manifest has no layer names"))?
        .iter()
        .filter_map(|v| v.as_str().map(str::to_owned))
        .collect();
    ensure!(
        names.len() == 34,
        "{} layer references, want stem + 32 blocks + ln_post",
        names.len()
    );

    let (mel_shape, mel) = f16_bits(&env.refs.join("layers/mel.npy"))?;
    let batch = mel_shape[0];
    ensure!(mel_shape == [batch, 128, 3000], "mel shape {mel_shape:?}");
    let n = batch * N_CTX * D_MODEL;
    let Env {
        gpu,
        blas,
        encoder,
        refs,
        ..
    } = &mut env;
    let mut ws = Workspace::new(gpu, batch)?;
    let mel_d = gpu.upload(mel)?;
    let out = gpu.upload(vec![NAN16; n])?;
    let load = |kind: &str, name: &str| -> Result<Vec<f64>> {
        let v = f64s(&refs.join(format!("layers/{kind}_{name}.npy")))?;
        ensure!(v.len() == n, "{kind}_{name}: {} values, want {n}", v.len());
        Ok(v)
    };

    // isolated: each layer on torch FP16's previous output
    let mut iso = Vec::new();
    let mut prev: Option<Vec<u16>> = None;
    for name in &names {
        let (_, torch16) = f16_bits(&refs.join(format!("layers/f16_{name}.npy")))?;
        ensure!(torch16.len() == n, "f16_{name}: {} values", torch16.len());
        gpu.upload_to(ws.x().ptr(), bytemuck::cast_slice(&vec![NAN16; n]))?;
        let ours = match (name.as_str(), &prev) {
            ("stem", _) => {
                encoder.stem(gpu, blas, mel_d.ptr(), &mut ws)?;
                download(gpu, ws.x(), n)?
            }
            ("ln_post", Some(p)) => {
                gpu.upload_to(ws.x().ptr(), bytemuck::cast_slice(p))?;
                gpu.upload_to(out.ptr(), bytemuck::cast_slice(&vec![NAN16; n]))?;
                encoder.ln_post(gpu, &ws, out.ptr())?;
                download(gpu, &out, n)?
            }
            (block, Some(p)) => {
                let i: usize = block.trim_start_matches("block").parse()?;
                gpu.upload_to(ws.x().ptr(), bytemuck::cast_slice(p))?;
                encoder.block(gpu, blas, i, &mut ws)?;
                download(gpu, ws.x(), n)?
            }
            (other, None) => bail!("layer {other} before the stem"),
        };
        iso.push(Row::new(
            name,
            &ours,
            &bits_f64(&torch16),
            &load("iso", name)?,
        )?);
        prev = Some(torch16);
    }

    // chained from the mel, the same kernels one layer at a time
    let (mut chain, mut noise, mut noise_conv) = (Vec::new(), Vec::new(), Vec::new());
    let mut last = Vec::new();
    for (k, name) in names.iter().enumerate() {
        let ours = match name.as_str() {
            "stem" => {
                encoder.stem(gpu, blas, mel_d.ptr(), &mut ws)?;
                download(gpu, ws.x(), n)?
            }
            "ln_post" => {
                gpu.upload_to(out.ptr(), bytemuck::cast_slice(&vec![NAN16; n]))?;
                encoder.ln_post(gpu, &ws, out.ptr())?;
                download(gpu, &out, n)?
            }
            _ => {
                encoder.block(gpu, blas, k - 1, &mut ws)?;
                download(gpu, ws.x(), n)?
            }
        };
        let (torch16, truth) = (load("f16", name)?, load("f32", name)?);
        let eff = Row::new(name, &load("f16eff", name)?, &torch16, &truth)?;
        let conv = Row::new(name, &load("f16gemm", name)?, &torch16, &truth)?;
        chain.push(Row::new(name, &ours, &torch16, &truth)?.bar_from(&[&eff, &conv]));
        noise.push(eff);
        noise_conv.push(conv);
        last = ours;
    }

    // the fused forward pass gives the chained per-layer result bit for bit
    gpu.upload_to(out.ptr(), bytemuck::cast_slice(&vec![NAN16; n]))?;
    encoder.forward(gpu, blas, mel_d.ptr(), out.ptr(), &mut ws)?;
    let fused = download(gpu, &out, n)?;
    ensure!(fused == last, "forward() differs from the per-layer chain");

    let mut failed = report(
        "isolated layers (truth: FP32 layer on torch FP16's input)",
        &iso,
        true,
    );
    failed.extend(report(
        "chained from the mel (truth: FP32 chain)",
        &chain,
        true,
    ));
    // the variants' own tables, printed for reading the chained bar, not asserted
    report(
        "noise yardstick, not asserted: torch FP16 efficient SDPA in the 'ours' columns, chained",
        &noise,
        false,
    );
    report(
        "noise yardstick, not asserted: torch FP16 with im2col-GEMM convs in the 'ours' columns, chained",
        &noise_conv,
        false,
    );
    ensure!(
        failed.is_empty(),
        "layers worse than {MAX_RATIO}x their torch FP16 bar: {failed:?}"
    );
    Ok(())
}

#[test]
fn encoder_end_to_end_matches_torch() -> Result<()> {
    let Some(env) = env()? else { return Ok(()) };
    let cases = env.manifest["e2e"].as_array().cloned().unwrap_or_default();
    ensure!(
        cases.len() >= 2,
        "{} end-to-end cases in the manifest",
        cases.len()
    );
    let Env {
        gpu,
        blas,
        encoder,
        refs,
        ..
    } = &env;

    let mut rows = Vec::new();
    for case in &cases {
        let (set, stem, seq) = (
            case["trace"].as_str().unwrap_or_default(),
            case["stem"].as_str().unwrap_or_default(),
            case["seq"].as_u64().unwrap_or_default(),
        );
        let traces = TraceSet::open(set).ok_or_else(|| eyre!("trace set {set} missing"))?;
        let call = traces
            .calls(stem, "encoder")?
            .into_iter()
            .find(|e| e.seq == seq)
            .ok_or_else(|| eyre!("{set}/{stem}: no encoder call #{seq}"))?;
        let mel = traces.array(call.arg("x")?)?;
        let recorded = traces.array(call.ret()?)?;
        let batch = mel.shape[0];
        ensure!(mel.shape == [batch, 128, 3000], "mel shape {:?}", mel.shape);
        ensure!(
            recorded.shape == [batch, N_CTX, D_MODEL],
            "output shape {:?}",
            recorded.shape
        );
        let ArrayData::F16(mel) = mel.data else {
            bail!("mel is not float16")
        };
        let n = batch * N_CTX * D_MODEL;

        let mut ws = Workspace::new(gpu, batch)?;
        let mel_d = gpu.upload(mel.iter().map(|h| h.to_bits()).collect::<Vec<u16>>())?;
        let out = gpu.upload(vec![NAN16; n])?;
        encoder.forward(gpu, blas, mel_d.ptr(), out.ptr(), &mut ws)?;
        let ours = download(gpu, &out, n)?;
        let torch16: Vec<f64> = recorded.to_f32()?.into_iter().map(f64::from).collect();
        let truth = f64s(&refs.join(case["f32"].as_str().unwrap_or_default()))?;
        let name = format!("{stem}#{seq}");
        let direct = Err::of(&ours, &torch16)?;
        println!(
            "{set}/{name} B={batch}: ours vs torch FP16 max {:.4e} mean {:.4e}",
            direct.max_abs, direct.mean_abs
        );
        rows.push(Row::new(&name, &ours, &torch16, &truth)?);
    }

    let failed = report("end to end (truth: FP32 encoder)", &rows, true);
    ensure!(
        failed.is_empty(),
        "worse than {MAX_RATIO}x torch FP16: {failed:?}"
    );
    Ok(())
}
