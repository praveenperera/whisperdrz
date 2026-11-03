//! A minimal RIFF/WAVE reader for 16-bit PCM, matching how the Python bench loads audio
//! (`soundfile.read(dtype="float32", always_2d=True).mean(axis=1)`)

use std::path::Path;

use eyre::{Context, Result, bail, ensure};

/// The samples of a 16-bit PCM wav at `rate` Hz, averaged over channels
pub fn read_mono(path: &Path, rate: u32) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).wrap_err_with(|| format!("read {}", path.display()))?;
    decode(&bytes, rate).wrap_err_with(|| format!("decode {}", path.display()))
}

fn decode(bytes: &[u8], rate: u32) -> Result<Vec<f32>> {
    ensure!(
        bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "not a RIFF/WAVE file"
    );

    let mut format = None;
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into()?) as usize;
        let body = &bytes[pos + 8..(pos + 8 + len).min(bytes.len())];
        match id {
            b"fmt " => {
                ensure!(body.len() >= 16, "short fmt chunk");
                let tag = u16::from_le_bytes(body[0..2].try_into()?);
                let channels = u16::from_le_bytes(body[2..4].try_into()?) as usize;
                let sample_rate = u32::from_le_bytes(body[4..8].try_into()?);
                let bits = u16::from_le_bytes(body[14..16].try_into()?);
                // 0xFFFE is WAVE_FORMAT_EXTENSIBLE, whose subformat we take as PCM given
                // the 16-bit check
                ensure!(tag == 1 || tag == 0xFFFE, "format tag {tag} is not PCM");
                ensure!(bits == 16, "{bits}-bit samples, want 16");
                ensure!(
                    sample_rate == rate,
                    "sample rate {sample_rate}, want {rate}"
                );
                ensure!(channels > 0, "no channels");
                format = Some(channels);
            }
            b"data" => {
                let Some(channels) = format else {
                    bail!("data chunk before fmt chunk")
                };
                ensure!(body.len() == len, "data chunk truncated");
                return Ok(mix(body, channels));
            }
            _ => {}
        }
        // chunks are padded to even sizes
        pos += 8 + len + (len & 1);
    }
    bail!("no data chunk")
}

/// soundfile scales 16-bit PCM by 1/32768 into float32, and numpy's float32 mean sums
/// then divides, which for one channel leaves the samples unchanged
fn mix(data: &[u8], channels: usize) -> Vec<f32> {
    data.chunks_exact(2 * channels)
        .map(|frame| {
            let sum: f32 = frame
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&s| i16::from_le_bytes(s) as f32 / 32768.0)
                .sum();
            sum / channels as f32
        })
        .collect()
}
