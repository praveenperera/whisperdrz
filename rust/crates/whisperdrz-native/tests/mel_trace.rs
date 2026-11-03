//! The log-mel port against torch: the window and filterbank against
//! `tests/data/mel/ref.json` (from `tools/gen_mel_ref.py`), then every recorded `mel`
//! call in the reference traces (`short3`, `e21c600`), which skip when missing

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use eyre::{Result, ensure, eyre};
use half::f16;
use serde_json::Value;
use whisperdrz_fixtures::TraceSet;
use whisperdrz_native::mel::{LogMel, N_FREQS, N_MELS};

fn f32_field(doc: &Value, key: &str) -> Result<Vec<f32>> {
    let b64 = doc[key]["b64"]
        .as_str()
        .ok_or_else(|| eyre!("ref.json has no {key}"))?;
    let bytes = STANDARD.decode(b64)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

#[test]
fn window_and_filterbank_match_torch() -> Result<()> {
    let doc: Value = serde_json::from_str(include_str!("data/mel/ref.json"))?;
    let mel = LogMel::new();

    let fb = f32_field(&doc, "fb")?;
    ensure!(
        fb.len() == N_FREQS * N_MELS,
        "ref fb has {} values",
        fb.len()
    );
    let fb_diff = fb
        .iter()
        .zip(mel.filterbank())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    ensure!(
        fb_diff == 0,
        "{fb_diff} filterbank values differ from torchaudio's bits"
    );

    // torch's vectorized cosf is a 1-ulp implementation, not correctly rounded; through
    // `0.5 - 0.5 cos` that is at most one ulp of a window value in [0.5, 1), 2^-24
    // absolute, however small the value
    let window = f32_field(&doc, "window")?;
    ensure!(
        window.len() == 400,
        "ref window has {} values",
        window.len()
    );
    let worst = window
        .iter()
        .zip(mel.window())
        .map(|(&a, &b)| (a as f64 - b as f64).abs())
        .fold(0.0, f64::max);
    let off = window
        .iter()
        .zip(mel.window())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    eprintln!("window: {off} of 400 values differ from torch, by at most {worst:.3e}");
    ensure!(
        worst <= 2f64.powi(-24),
        "window differs from torch by {worst:.3e}"
    );
    Ok(())
}

#[test]
fn mel_matches_recorded_calls() -> Result<()> {
    let mel = LogMel::new();
    let mut calls = 0;
    for name in ["short3", "e21c600"] {
        let Some(set) = TraceSet::open(name) else {
            continue;
        };
        for stem in set.stems()? {
            for ev in set.calls(&stem, "mel")? {
                let wave = set.array(ev.arg("waveform")?)?;
                let want = set.array(ev.ret()?)?;
                ensure!(
                    wave.shape.len() == 2 && wave.shape[0] == 1,
                    "waveform shape {:?}",
                    wave.shape
                );
                let audio = wave.to_f32()?;
                let got = mel.compute(&audio)?;
                let frames = LogMel::frames(audio.len());
                ensure!(
                    want.shape == [1, N_MELS, frames],
                    "{stem} #{}: recorded mel {:?}, ours [1, {N_MELS}, {frames}]",
                    ev.seq,
                    want.shape
                );
                let want = want.to_f32()?;
                ensure!(got.iter().all(|v| v.is_finite()), "non-finite mel");

                let (mut max_abs, mut sum_abs, mut f16_diff) = (0f64, 0f64, 0usize);
                for (&a, &b) in got.iter().zip(&want) {
                    let d = (a as f64 - b as f64).abs();
                    max_abs = max_abs.max(d);
                    sum_abs += d;
                    f16_diff += usize::from(f16::from_f32(a) != f16::from_f32(b));
                }

                let n = want.len() as f64;
                eprintln!(
                    "{name}/{stem} #{}: max-abs {max_abs:.3e}, mean-abs {:.3e}, fp16 differ {:.4}% ({f16_diff} of {})",
                    ev.seq,
                    sum_abs / n,
                    100.0 * f16_diff as f64 / n,
                    want.len()
                );
                // FP32 FFT and sgemm noise on values of order 1 sits near 1e-6; a
                // systematic error (window, filterbank, padding, framing) is far larger
                ensure!(
                    max_abs < 1e-4,
                    "{name}/{stem} #{}: max-abs {max_abs:.3e}",
                    ev.seq
                );
                calls += 1;
            }
        }
    }

    if calls == 0 {
        eprintln!("skipping: no traces");
        return Ok(());
    }

    ensure!(calls == 35, "checked {calls} mel calls, the traces hold 35");
    Ok(())
}
