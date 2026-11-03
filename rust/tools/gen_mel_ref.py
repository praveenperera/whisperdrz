"""Generate the mel port's committed test data by running the Python reference on the CPU

Writes `crates/whisperdrz-native/tests/data/mel/ref.json`: the Hann window and the
torchaudio slaney filterbank exactly as `LogMelSpectrogram` builds them, as little-endian
FP32 bits, plus the linspace and mel-point intermediates, so the Rust port can be checked
bit for bit before any audio goes through it

Run from `rust/` with the snapshot source first on the path, on the CPU only:
  CUDA_VISIBLE_DEVICES="" ~/code/whisperdrz/.venv/bin/python -I tools/gen_mel_ref.py SRC_DIR
"""

import base64
import json
import os
import sys

SRC = sys.argv[1]
sys.path.insert(0, SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchaudio.functional as AF  # noqa: E402

from whisperdrz.audio import LogMelSpectrogram  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "crates", "whisperdrz-native", "tests", "data", "mel")


def f32_b64(t):
    a = np.ascontiguousarray(t.detach().cpu().numpy(), dtype="<f4")
    return {"shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode()}


def main():
    m = LogMelSpectrogram(n_mels=128)
    sr, n_mels, n_freqs = 16000, 128, 201
    m_min = AF.functional._hz_to_mel(0.0, mel_scale="slaney")
    m_max = AF.functional._hz_to_mel(float(sr // 2), mel_scale="slaney")
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    out = {
        "window": f32_b64(m.window),
        "fb": f32_b64(m.mel_scale.fb),
        "all_freqs": f32_b64(torch.linspace(0, sr // 2, n_freqs)),
        "m_max": m_max,
        "m_pts": f32_b64(m_pts),
        "f_pts": f32_b64(AF.functional._mel_to_hz(m_pts.clone(), mel_scale="slaney")),
        "torch": torch.__version__,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ref.json"), "w") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")


if __name__ == "__main__":
    main()
