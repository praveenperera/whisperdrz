//! whisperdrz native CLI: runs the reference bench on the native GPU runtime
//!
//!   whisperdrz-native bench LIST OUT_JSON [--weights FILE] [--ecapa FILE]
//!
//! LIST holds one 16 kHz wav path per line. The run follows the Python bench
//! (`_scratch/native-port/bench.py` with `--pad-fix`): one transcriber for all files, an
//! untimed warmup on the first 60 s of the first file, and the speaker-crop RNG reseeded
//! to 0 before each call. Files run in list order, since transcriber state carries
//! across calls, and OUT_JSON maps each basename to `{text, segments}` sorted by name,
//! the order `rust/gate/gate.py` reads

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use eyre::{Context, Result, bail, ensure};
use serde_json::{Map, Value, json};
use whisperdrz_native::cuda::model::GpuModel;
use whisperdrz_native::decode::tables::DecoderTables;
use whisperdrz_native::mel::{LogMel, N_MELS, SAMPLE_RATE};
use whisperdrz_native::speaker::PyRandom;
use whisperdrz_native::text::Tokenizer;
use whisperdrz_native::transcriber::{Config, Mel, Transcriber};

mod wav;

/// Seconds of the first file the warmup call transcribes
const WARMUP_SECONDS: usize = 60;

struct Args {
    list: PathBuf,
    out: PathBuf,
    weights: PathBuf,
    ecapa: PathBuf,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let usage = "usage: whisperdrz-native bench LIST OUT_JSON [--weights FILE] [--ecapa FILE]";
        ensure!(args.next().as_deref() == Some("bench"), "{usage}");
        let (Some(list), Some(out)) = (args.next(), args.next()) else {
            bail!("{usage}")
        };

        let mut weights = None;
        let mut ecapa = None;
        while let Some(flag) = args.next() {
            let value = args.next().map(PathBuf::from);
            match flag.as_str() {
                "--weights" => weights = value,
                "--ecapa" => ecapa = value,
                _ => bail!("unknown flag {flag}\n{usage}"),
            }
        }

        Ok(Self {
            list: list.into(),
            out: out.into(),
            weights: match weights {
                Some(w) => w,
                None => default_weights()?,
            },
            ecapa: ecapa
                .unwrap_or_else(|| home().join(".cache/whisperdrz/ecapa2/ecapa2.safetensors")),
        })
    }
}

fn home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default()
}

/// The FP16 checkpoint in the Hugging Face cache, as `scripts/gpu-test.sh` finds it
fn default_weights() -> Result<PathBuf> {
    let snapshots = home().join(".cache/huggingface/hub/models--fluxions--whisperdrz/snapshots");
    let mut found: Vec<PathBuf> = std::fs::read_dir(&snapshots)
        .wrap_err_with(|| format!("read {}", snapshots.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path().join("whisperdrz-large-v3.safetensors"))
        .filter(|p| p.is_file())
        .collect();
    found.sort();
    found.into_iter().next().ok_or_else(|| {
        eyre::eyre!(
            "no checkpoint under {} (pass --weights)",
            snapshots.display()
        )
    })
}

fn basename(path: &Path) -> Result<String> {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(str::to_owned)
        .ok_or_else(|| eyre::eyre!("no file name in {}", path.display()))
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Args::parse()?;

    let paths: Vec<PathBuf> = std::fs::read_to_string(&args.list)
        .wrap_err_with(|| format!("read {}", args.list.display()))?
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(PathBuf::from)
        .collect();
    ensure!(!paths.is_empty(), "{} lists no files", args.list.display());

    let audios = paths
        .iter()
        .map(|p| wav::read_mono(p, SAMPLE_RATE as u32))
        .collect::<Result<Vec<_>>>()?;

    let config = Config::default();
    let model = GpuModel::new(&args.weights, &args.ecapa, config.bs, config.bs)?;
    let tables = Arc::new(DecoderTables::new(Tokenizer::multilingual()?)?);
    let log_mel = LogMel::new();
    let mel = Box::new(move |audio: &[f32]| {
        let data = log_mel.compute(audio)?;
        Mel::new(N_MELS, data.len() / N_MELS, data)
    });
    let mut trn = Transcriber::new(config, model, mel, tables)?;

    let warmup = &audios[0][..audios[0].len().min(WARMUP_SECONDS * SAMPLE_RATE)];
    trn.transcribe(warmup, &mut PyRandom::new(0))?;

    let mut out = Map::new();
    let mut total_audio = 0.0;
    let mut total_wall = 0.0;
    for (path, audio) in paths.iter().zip(&audios) {
        let start = Instant::now();
        let result = trn.transcribe(audio, &mut PyRandom::new(0))?;
        let wall = start.elapsed().as_secs_f64();

        let seconds = audio.len() as f64 / SAMPLE_RATE as f64;
        total_audio += seconds;
        total_wall += wall;
        let name = basename(path)?;
        println!(
            "{name} {seconds:.1}s audio {wall:.3}s wall {:.1}x",
            seconds / wall
        );
        out.insert(
            name,
            json!({ "text": result.text, "segments": serde_json::to_value(&result.segments)? }),
        );
    }

    println!(
        "TOTAL {} files {total_audio:.1}s audio {total_wall:.3}s wall RTFx {:.2}",
        paths.len(),
        total_audio / total_wall
    );
    // serde_json's default map is sorted, the reference JSON's file order
    std::fs::write(&args.out, serde_json::to_string(&Value::Object(out))?)
        .wrap_err_with(|| format!("write {}", args.out.display()))?;
    Ok(())
}
