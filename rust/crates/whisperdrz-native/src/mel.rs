//! Whisper's log-mel front end on the CPU, as `LogMelSpectrogram` in `audio.py` computes
//! it for the transcriber
//!
//! The reference runs on the CPU in FP32: `torch.stft` (n_fft 400, hop 160, periodic Hann
//! window, centered with reflect padding), drop the last frame, power `|X|^2`, the
//! torchaudio slaney mel filterbank, `log10(max(x, 1e-10))`, `max(x, max - 8)` over the
//! whole input and `(x + 4) / 4`. The transcriber then casts the result to FP16 on the GPU.
//!
//! The window and filterbank are built with torch's FP32 operation order, so the
//! filterbank matches torchaudio bit for bit. The transform, power and filterbank product
//! run in f64 and round once to FP32 before the log, which keeps this port within FP32
//! noise of torch's FP32 FFT and sgemm

use std::sync::Arc;

use eyre::{Result, ensure};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 128;
/// One-sided frequency bins of an `N_FFT` transform
pub const N_FREQS: usize = N_FFT / 2 + 1;
/// Frames one parallel log-mel task computes; a 30 s chunk has 3001
const FRAMES_PER_TASK: usize = 128;

/// The log-mel transform with its window, filterbank and FFT plan
pub struct LogMel {
    window: Vec<f32>,
    /// `[N_FREQS][N_MELS]`, torchaudio's `MelScale.fb`
    fb: Vec<f32>,
    /// Per mel band, the first and one-past-last frequency bin with a nonzero weight;
    /// the product skips the zeros, which add exactly nothing
    bands: Vec<(usize, usize)>,
    fft: Arc<dyn RealToComplex<f64>>,
}

impl Default for LogMel {
    fn default() -> Self {
        Self::new()
    }
}

impl LogMel {
    pub fn new() -> Self {
        let fb = slaney_filterbank();
        let bands = (0..N_MELS)
            .map(|m| {
                let nonzero = |f: &usize| fb[f * N_MELS + m] != 0.0;
                let lo = (0..N_FREQS).find(nonzero).unwrap_or(0);
                let hi = (0..N_FREQS).rev().find(nonzero).map_or(lo, |f| f + 1);
                (lo, hi)
            })
            .collect();
        Self {
            window: hann_window(N_FFT),
            fb,
            bands,
            fft: RealFftPlanner::<f64>::new().plan_fft_forward(N_FFT),
        }
    }

    /// The periodic Hann window, `torch.hann_window(400)`
    pub fn window(&self) -> &[f32] {
        &self.window
    }

    /// The `[N_FREQS][N_MELS]` filterbank, torchaudio's `MelScale(..., norm="slaney",
    /// mel_scale="slaney").fb`
    pub fn filterbank(&self) -> &[f32] {
        &self.fb
    }

    /// Frames for `n_samples` of audio: the centered STFT gives `1 + n / hop` and the
    /// reference drops the last
    pub fn frames(n_samples: usize) -> usize {
        n_samples / HOP_LENGTH
    }

    /// The normalized log-mel spectrogram of 16 kHz mono audio, `[N_MELS][frames]`
    /// row-major, the `(n_mels, frames)` tensor the reference produces
    pub fn compute(&self, audio: &[f32]) -> Result<Vec<f32>> {
        // torch's reflect padding needs more samples than the pad on each side
        ensure!(
            audio.len() > N_FFT / 2,
            "log-mel needs more than {} samples, got {}",
            N_FFT / 2,
            audio.len()
        );
        let frames = Self::frames(audio.len());
        // frames are independent until the global floor below, so blocks of them run in
        // parallel, each with its own FFT buffers, into a frame-major buffer
        let mut by_frame = vec![0f32; frames * N_MELS];
        by_frame
            .par_chunks_mut(FRAMES_PER_TASK * N_MELS)
            .enumerate()
            .for_each(|(block, out)| {
                let mut input = self.fft.make_input_vec();
                let mut spectrum = self.fft.make_output_vec();
                let mut scratch = self.fft.make_scratch_vec();
                let mut power = vec![0f64; N_FREQS];
                for (k, row) in out.as_chunks_mut::<N_MELS>().0.iter_mut().enumerate() {
                    let t = block * FRAMES_PER_TASK + k;
                    let start = (t * HOP_LENGTH) as isize - (N_FFT / 2) as isize;
                    for (n, slot) in input.iter_mut().enumerate() {
                        let x = audio[reflect(start + n as isize, audio.len())];
                        *slot = x as f64 * self.window[n] as f64;
                    }

                    self.fft
                        .process_with_scratch(&mut input, &mut spectrum, &mut scratch)
                        .expect("FFT buffers come from the plan");
                    for (p, Complex { re, im }) in power.iter_mut().zip(&spectrum) {
                        *p = re * re + im * im;
                    }

                    for (m, &(lo, hi)) in self.bands.iter().enumerate() {
                        let sum: f64 = (lo..hi)
                            .map(|f| power[f] * self.fb[f * N_MELS + m] as f64)
                            .sum();
                        row[m] = sum.max(1e-10).log10() as f32;
                    }
                }
            });

        let mut mel = vec![0f32; N_MELS * frames];
        for (t, row) in by_frame.as_chunks::<N_MELS>().0.iter().enumerate() {
            for (m, &v) in row.iter().enumerate() {
                mel[m * frames + t] = v;
            }
        }

        let floor = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max) - 8.0;
        for v in &mut mel {
            *v = (v.max(floor) + 4.0) / 4.0;
        }

        Ok(mel)
    }
}

/// Index of padded position `i` under reflect padding (no edge repeat) of `len` samples
fn reflect(i: isize, len: usize) -> usize {
    let last = len as isize - 1;
    let i = i.abs();
    (if i > last { 2 * last - i } else { i }) as usize
}

/// `torch.hann_window(n)` (periodic) in torch's FP32 order: `cos(k * (2 pi / n))`, then
/// `* -0.5 + 0.5`, each rounded to FP32
///
/// torch's vectorized `cosf` is accurate to 1 ulp rather than correctly rounded, so 12 of
/// the 400 values differ from these, by at most 2^-24
fn hann_window(n: usize) -> Vec<f32> {
    let step = (std::f64::consts::TAU / n as f64) as f32;
    (0..n)
        .map(|k| {
            let c = ((k as f32 * step) as f64).cos() as f32;
            c * -0.5 + 0.5
        })
        .collect()
}

/// The `N_MELS + 2` slaney mel points in Hz exactly as torchaudio's FP32
/// `_mel_to_hz(torch.linspace(0, hz_to_mel(8000), 130))` gives them, as FP32 bits
///
/// torch's vectorized `expf` is accurate to 1 ulp rather than correctly rounded, which
/// moves 2 of these 130 points by 1 ulp from any correctly rounded evaluation and with
/// them 27 filterbank weights, so the points are torch's own values (from
/// `tools/gen_mel_ref.py`); `tests::mel_points_follow_the_slaney_scale` rederives them
const MEL_POINTS_HZ: [u32; N_MELS + 2] = [
    0x00000000, 0x41bb0fe4, 0x423b0fe4, 0x428c4beb, 0x42bb0fe4, 0x42e9d3dc, 0x430c4beb, 0x4323ade7,
    0x433b0fe4, 0x435271e0, 0x4369d3dc, 0x43809aec, 0x438c4beb, 0x4397fce9, 0x43a3ade7, 0x43af5ee5,
    0x43bb0fe4, 0x43c6c0e2, 0x43d271e0, 0x43de22de, 0x43e9d3dc, 0x43f584db, 0x44009aec, 0x4406736c,
    0x440c4beb, 0x4412246a, 0x4417fce9, 0x441dd568, 0x4423ade7, 0x44298666, 0x442f5ee5, 0x44353764,
    0x443b0fe4, 0x4440e863, 0x4446c0e2, 0x444c9961, 0x445271e0, 0x44584a5f, 0x445e22de, 0x4463fb5d,
    0x4469d3dc, 0x446fac5c, 0x447584db, 0x447b694c, 0x4480c616, 0x4483eab3, 0x448722f2, 0x448a6f4f,
    0x448dd048, 0x4491465f, 0x4494d214, 0x449873f2, 0x449c2c83, 0x449ffc53, 0x44a3e3f4, 0x44a7e3fa,
    0x44abfd00, 0x44b02fa0, 0x44b47c7a, 0x44b8e432, 0x44bd6771, 0x44c206e3, 0x44c6c336, 0x44cb9d20,
    0x44d0955b, 0x44d5aca3, 0x44dae3ba, 0x44e03b67, 0x44e5b474, 0x44eb4fb5, 0x44f10dff, 0x44f6f029,
    0x44fcf714, 0x450191d7, 0x4504bb6c, 0x4507f8c3, 0x450b4a57, 0x450eb0ab, 0x45122c3b, 0x4515bd8d,
    0x45196529, 0x451d239e, 0x4520f976, 0x4524e743, 0x4528ed9e, 0x452d0d1f, 0x45314666, 0x45359a0c,
    0x453a08bd, 0x453e9320, 0x454339e2, 0x4547fdb2, 0x454cdf49, 0x4551df63, 0x4556feb9, 0x455c3e11,
    0x45619e2e, 0x45671fe5, 0x456cc408, 0x45728b63, 0x457876dd, 0x457e8755, 0x45825ed8, 0x45858d70,
    0x4588cfe6, 0x458c26bc, 0x458f9270, 0x45931380, 0x4596aa78, 0x459a57de, 0x459e1c3f, 0x45a1f829,
    0x45a5ec2c, 0x45a9f8e7, 0x45ae1eee, 0x45b25ede, 0x45b6b961, 0x45bb2f16, 0x45bfc0a8, 0x45c46ec6,
    0x45c93a1f, 0x45ce2372, 0x45d32b74, 0x45d852e1, 0x45dd9a87, 0x45e30329, 0x45e88d97, 0x45ee3aa4,
    0x45f40b24, 0x45f9fffc,
];

/// `torch.linspace(start, end, steps)` in FP32: the first half counts up from `start`,
/// the rest down from `end`, with an FP32 step
fn linspace(start: f32, end: f32, steps: usize) -> Vec<f32> {
    let step = (end - start) / (steps - 1) as f32;
    let halfway = steps / 2;
    (0..steps)
        .map(|i| {
            if i < halfway {
                start + step * i as f32
            } else {
                end - step * (steps - i - 1) as f32
            }
        })
        .collect()
}

/// torchaudio's `melscale_fbanks(201, 0, 8000, 128, 16000, "slaney", "slaney")` in its
/// FP32 operation order: triangles between consecutive mel points, area-normalized,
/// `[N_FREQS][N_MELS]`
fn slaney_filterbank() -> Vec<f32> {
    let all_freqs = linspace(0.0, (SAMPLE_RATE / 2) as f32, N_FREQS);
    let f_pts = MEL_POINTS_HZ.map(f32::from_bits);
    let f_diff: Vec<f32> = f_pts.windows(2).map(|w| w[1] - w[0]).collect();

    let mut fb = vec![0f32; N_FREQS * N_MELS];
    for (f, &freq) in all_freqs.iter().enumerate() {
        for m in 0..N_MELS {
            let down = -(f_pts[m] - freq) / f_diff[m];
            let up = (f_pts[m + 2] - freq) / f_diff[m + 1];
            let enorm = 2.0 / (f_pts[m + 2] - f_pts[m]);
            // torch's max(0, min(down, up)) as plain compares, which keep the -0.0 that
            // torchaudio's first weight carries; f32::max may return either zero
            let tri = if up < down { up } else { down };
            let tri = if 0.0 > tri { 0.0 } else { tri };
            fb[f * N_MELS + m] = tri * enorm;
        }
    }

    fb
}

#[cfg(test)]
mod tests {
    use super::{LogMel, MEL_POINTS_HZ, N_MELS, SAMPLE_RATE, linspace, reflect};

    #[test]
    fn reflect_padding() {
        let idx: Vec<usize> = (-3..8).map(|i| reflect(i, 5)).collect();
        assert_eq!(idx, [3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1]);
    }

    /// torchaudio's `_hz_to_mel` and `_mel_to_hz` (slaney) on torch's FP32 linspace, with
    /// a correctly rounded exp: each point within 1 ulp of the table, so the table is the
    /// slaney scale and only torch's exp rounding sets its last bits
    #[test]
    fn mel_points_follow_the_slaney_scale() {
        let f_sp = 200.0 / 3.0;
        let min_log_mel = 1000.0 / f_sp;
        let logstep = 6.4f64.ln() / 27.0;
        let m_max = min_log_mel + ((SAMPLE_RATE / 2) as f64 / 1000.0).ln() / logstep;
        let mut off = 0;
        for (mel, &bits) in linspace(0.0, m_max as f32, N_MELS + 2)
            .iter()
            .zip(&MEL_POINTS_HZ)
        {
            let hz = if *mel as f64 >= min_log_mel {
                let x = (mel - min_log_mel as f32) * logstep as f32;
                1000.0 * (x as f64).exp() as f32
            } else {
                f_sp as f32 * mel
            };
            let ulps = (hz.to_bits() as i64 - bits as i64).abs();
            assert!(
                ulps <= 1,
                "mel {mel}: {hz} is {ulps} ulp from torch's {}",
                f32::from_bits(bits)
            );
            off += usize::from(ulps != 0);
        }
        assert_eq!(off, 2, "torch's expf rounding moves exactly 2 points");
    }

    #[test]
    fn short_input_is_an_error() {
        let mel = LogMel::new();
        assert!(mel.compute(&[0.0; 200]).is_err());
        assert_eq!(mel.compute(&[0.0; 16_000]).unwrap().len(), N_MELS * 100);
    }
}
