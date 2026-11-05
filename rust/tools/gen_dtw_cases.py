"""Generate the DTW port's committed test data by running the Python reference on the CPU

Writes `crates/whisperdrz-native/tests/data/dtw/`:
  tokenizer.json   special ids and `encode(ch)` for every char the tests and the
                   reference traces' DTW window texts use
  cases.json       expected values for the DTW kernel, anchor bands, parsing,
                   chunking, the text rebuild and full `fill_word_timings_batch`
                   runs with the GPU forward replaced by recorded synthetic matrices

Run from `rust/` with the snapshot source first on the path, on the CPU only:
  CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 \
    ~/code/whisperdrz/.venv/bin/python -I tools/gen_dtw_cases.py SRC_DIR [FIXTURES_DIR]
"""

import base64
import json
import os
import sys

SRC = sys.argv[1]
FIXTURES = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/.cache/whisperdrz/fixtures/ref-a783291")
sys.path.insert(0, SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import whisperdrz.dtw_align as dtw  # noqa: E402
from whisperdrz.tokenizer import get_tokenizer  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "crates", "whisperdrz-native", "tests", "data", "dtw")
TOK = get_tokenizer(True, num_languages=100, language="en")
RNG = np.random.default_rng(20261103)


def f32_b64(a):
    a = np.ascontiguousarray(a, dtype="<f4")
    return {"shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode()}


def f16_b64(a):
    a = np.ascontiguousarray(a, dtype="<f2")
    return {"shape": list(a.shape), "f16": base64.b64encode(a.tobytes()).decode()}


def plain(x):
    """JSON-ready copy: tuples to lists, numpy scalars to Python, non-finite floats to strings"""
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return repr(x)
    return x


# ---------- texts ----------

TEXTS = [
    "\n[0] <|0.00|>I'm<|0.18|> <|0.24|>right<|0.36|> <|0.38|>here,<|0.62|> on CBS\n",
    "\n[0] <|0.00|>[tut]<|0.30|> <|0.70|>And<|0.74|>\n",
    "no speaker tag at all, just words <|1.20|>timed<|1.50|> here",
    "lead text [1] then <|0.50|>[breath]<|0.90|> word [c] crosstalk [07] same as seven [7] again",
    "[0] <accent:british>Hello<emotion:happy> there [CHOP] friend <|2.04|>[laugh]<|2.50|>\n[1] ok",
    "[0] half <|1.00|>timed and <|2.00|>open<|2.40|> <|2.50|>[pause]<|3.00|>\n[1] <|3.20|>next<|3.30|>",
    "[0] café naïve 日本語 \U0001f600 words with nbsp\x1cand fs <|4.10|>end<|4.30|>",
    "[0] [laugh] [/laugh] <|0.10|>[music]<|0.40|>\n[1] [breath]",
    "[0] broken <| tags [ and < here [12x] <|5.5|> and <|0.01|>tie<|0.03|> ok",
    "",
    "   ",
    "[0] <|0.00|>uh...<|29.98|>\n[1] <|29.60|>\n",
]


def long_text(n_lines, words_per_line, t0=0.0):
    """A multi-line window long enough to need several chunks at a small budget"""
    vocab = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike".split()
    parts = []
    t = t0
    k = 0
    for li in range(n_lines):
        parts.append(f"\n[{li % 3}] ")
        for wi in range(words_per_line):
            w = vocab[k % len(vocab)]
            k += 1
            if (k % 5) == 0:
                parts.append(f"<|{t:.2f}|>{w}<|{t + 0.14:.2f}|> ")
            elif (k % 11) == 0:
                parts.append(f"<|{t:.2f}|>[breath]<|{t + 0.2:.2f}|> {w} ")
            else:
                parts.append(f"{w} ")
            t += 0.31
    parts.append("\n")
    return "".join(parts)


LONG = [long_text(4, 9), long_text(2, 30, 1.0), long_text(6, 5, 0.5)]


# ---------- char table ----------


def trace_texts():
    texts = []
    for name in ("short3", "e21c600"):
        root = os.path.join(FIXTURES, name)
        summary = os.path.join(root, "summary.json")
        if not os.path.exists(summary):
            print(f"note: no trace set {root}, char table covers synthetic texts only", file=sys.stderr)
            continue
        for f in json.load(open(summary))["files"]:
            for line in open(os.path.join(root, f["stem"], "events.jsonl"), encoding="utf-8"):
                if '"fill_word_timings_batch"' not in line:
                    continue
                ev = json.loads(line)
                if ev["ev"] != "fill_word_timings_batch":
                    continue
                for slot in ev["args"]["slot_data"]:
                    texts.append(slot["__tuple"][1])
    return texts


def char_table(texts):
    chars = sorted({ch for t in texts for ch in t} | {" "})
    return {ch: list(TOK.encode(ch)) for ch in chars}


# ---------- kernel cases ----------


def random_matrix(n, m, kind):
    if kind == "normal":
        return RNG.standard_normal((n, m)).astype(np.float32)
    if kind == "ties":
        # small integers make equal costs common, which exercises the tie order
        return RNG.integers(-2, 3, size=(n, m)).astype(np.float32)
    if kind == "banded":
        x = RNG.standard_normal((n, m)).astype(np.float32)
        for _ in range(max(1, n // 3)):
            r = RNG.integers(0, n)
            c = RNG.integers(0, m)
            if RNG.random() < 0.5:
                x[r, :c] = np.inf
            else:
                x[r, c:] = np.inf
        return x
    if kind == "allinf":
        x = np.full((n, m), np.inf, dtype=np.float32)
        x[0, 0] = -1.0
        return x
    raise ValueError(kind)


def kernel_cases():
    out = []
    shapes = [(1, 1), (1, 7), (7, 1), (3, 3), (5, 40), (40, 5), (23, 61), (40, 120)]
    for n, m in shapes:
        for kind in ("normal", "ties", "banded", "allinf"):
            x = random_matrix(n, m, kind)
            ti, mi = dtw.dtw_cpu(x)
            out.append({"x": f32_b64(x), "text_idx": ti.tolist(), "time_idx": mi.tolist()})
    return out


def word_groups(n_rows):
    """Random token groups covering `n_rows - 1` rows plus the eot group, as the pipeline builds them"""
    groups = []
    left = n_rows - 1
    while left > 0:
        k = int(min(left, RNG.integers(1, 6)))
        groups.append([int(v) for v in RNG.integers(0, 1000, size=k)])
        left -= k
    words = [f"w{i}" for i in range(len(groups))]
    return [*words, ""], [*groups, [TOK.eot]]


def neg_matrix_cases():
    out = []
    for n, m in [(2, 5), (6, 30), (25, 80), (40, 120), (12, 4)]:
        for kind in ("normal", "ties", "banded"):
            x = random_matrix(n, m, kind)
            words, groups = word_groups(n)
            # a row count that disagrees with the groups hits the early returns
            for g in (groups, groups[:-2] + groups[-1:], [*groups[:-1], [1, 2, 3], groups[-1]]):
                w = words[: len(g)]
                ret = dtw._dtw_from_neg_matrix(x, w, g)
                out.append({"x": f32_b64(x), "words": w, "word_tokens": g, "ret": plain(ret)})
    out.append({"x": f32_b64(np.zeros((0, 4), np.float32)), "words": ["", ""], "word_tokens": [[1], [2]], "ret": []})
    out.append({"x": f32_b64(np.zeros((3, 3), np.float32)), "words": [""], "word_tokens": [[TOK.eot]], "ret": []})
    return out


def band_cases():
    out = []
    for n, m in [(10, 40), (30, 90), (40, 100), (5, 3)]:
        for trial in range(6):
            x = RNG.standard_normal((n, m)).astype(np.float32)
            words, groups = word_groups(n)
            nw = len(words) - 1
            audio_cap = int(m if trial % 2 == 0 else max(1, m - 5))
            secs = m / 50.0
            word_anchors = []
            for wi in sorted(RNG.choice(nw + 2, size=min(nw + 2, 4), replace=False).tolist()):
                a_s = round(float(RNG.uniform(-0.2, secs)), 2)
                a_e = round(a_s + float(RNG.uniform(-0.05, 0.6)), 2)
                word_anchors.append((int(wi), a_s, a_e))
            # half-way frames: 0.01 * 50 and 0.03 * 50 round half to even
            word_anchors.append((0, 0.01, 0.03))
            tag_anchors = []
            for _ in range(3):
                a_s = round(float(RNG.uniform(-0.3, secs + 0.3)), 2)
                a_e = round(a_s + float(RNG.uniform(0.0, 0.5)), 2)
                p = int(RNG.integers(0, nw + 2)) if RNG.random() < 0.8 else None
                q = int(RNG.integers(0, nw + 2)) if RNG.random() < 0.8 else None
                tag_anchors.append((a_s, a_e, p, q))
            slack = int(7 if trial < 4 else trial)
            after = x.copy()
            dtw._apply_anchor_bands(
                after,
                words,
                groups,
                audio_cap,
                word_anchors if trial != 3 else None,
                tag_anchors if trial != 2 else None,
                slack,
            )
            out.append(
                {
                    "x": f32_b64(x),
                    "words": words,
                    "word_tokens": groups,
                    "audio_cap": audio_cap,
                    "word_anchors": word_anchors if trial != 3 else [],
                    "tag_anchors": tag_anchors if trial != 2 else [],
                    "slack": slack,
                    "after": f32_b64(after),
                }
            )
    return out


# ---------- text cases ----------


def spk_json(spk):
    return {"c": spk} if isinstance(spk, str) else {"n": spk}


def parse_cases():
    out = []
    for text in [*TEXTS, *LONG]:
        lines, flat, tags = dtw.parse_predicted_lines(text)
        out.append(
            {
                "text": text,
                "lines": [[spk_json(spk), [list(w) for w in ws]] for spk, ws in lines],
                "flat": plain(flat),
                "tag_anchors": plain(tags),
            }
        )
    return out


def chunk_cases():
    out = []
    for text in [*TEXTS, *LONG]:
        lines, flat, _ = dtw.parse_predicted_lines(text)
        for max_tokens in (12, 40, 90, 440):
            chunks = dtw.chunk_speaker_lines(lines, TOK, max_tokens)
            out.append(
                {
                    "text": text,
                    "max_tokens": max_tokens,
                    "chunks": [[[spk_json(spk), [w[0] for w in ws]] for spk, ws in c] for c in chunks],
                    "rendered": [dtw._render_chunk(c)[0] for c in chunks],
                    "ids": [dtw.tokens_chars(dtw._render_chunk(c)[0], TOK)[0] for c in chunks],
                }
            )
    return out


def rebuild_cases():
    out = []
    for text in [*TEXTS, *LONG]:
        _, flat, _ = dtw.parse_predicted_lines(text)
        for trial in range(3):
            timings = []
            for i in range(len(flat)):
                if trial == 2 and i % 3 == 0:
                    timings.append(None)
                    continue
                s = float(RNG.uniform(-0.05, 30.0))
                # odd hundredths and exact half-frames test the 0.02 s quantisation
                if i % 4 == 1:
                    s = round(s, 2) + 0.01
                if i % 4 == 2:
                    s = round(s * 100) / 100 + 0.005
                timings.append((s, s + float(RNG.uniform(-0.03, 1.0))))
            out.append(
                {"text": text, "timings": plain(timings), "ret": dtw.rebuild_with_timings(text, flat, timings)}
            )
    return out


# ---------- full batch with the GPU part replaced ----------


class FakeModel:
    def __init__(self):
        self.tokens = None

        class Decoder:
            def __call__(inner, tokens, xa):  # noqa: N805
                self.tokens = tokens

        self.decoder = Decoder()


class FakeCapture:
    def __init__(self, model, layers=None):
        self.model = model

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def compute_attention(self, head_mask):
        t = self.model.tokens
        w = torch.zeros(t.shape[0], 1, t.shape[1], self.model.n_audio)
        return w[0] if w.shape[0] == 1 else w


def batch_cases():
    out = []
    orig_capture = dtw.CrossAttnCapture
    orig_prep = dtw._dtw_prep_batch
    dtw.CrossAttnCapture = FakeCapture
    calls = []

    def fake_prep(full_weights, actual_lens, audio_caps, medfilt_width=7):
        b, _, t_pad, _ = full_weights.shape
        cap = max(audio_caps)
        # f16-representable values keep the committed data small; the DTW still sums them in f32
        h = (RNG.standard_normal((b, t_pad, cap)) * 0.7).astype(np.float16)
        calls.append({"actual_lens": list(actual_lens), "audio_caps": list(audio_caps), "matrices": f16_b64(h)})
        m = h.astype(np.float32)
        return m

    dtw._dtw_prep_batch = fake_prep
    try:
        groups = [
            ([TEXTS[0]], [216], 1500, 440),
            ([TEXTS[1]], [74], 1500, 440),
            ([TEXTS[2], TEXTS[3], TEXTS[4]], [400, 300, 151], 1500, 440),
            ([TEXTS[5], TEXTS[6], TEXTS[7]], [300, 299, 0], 1500, 440),
            ([TEXTS[8], TEXTS[9], TEXTS[10]], [200, 200, 200], 1500, 440),
            ([TEXTS[11]], [3000], 1500, 440),
            ([LONG[0], LONG[1]], [300, 3000], 150, 60),
            ([LONG[2], TEXTS[3]], [250, 4000], 160, 40),
            ([LONG[1]], [-5], 1500, 50),
            ([TEXTS[9], TEXTS[10]], [3000, 3000], 1500, 440),
        ]
        for texts, frames, n_audio, max_tokens in groups:
            model = FakeModel()
            model.n_audio = n_audio
            slot_data = [(torch.zeros(n_audio, 1), t, nf) for t, nf in zip(texts, frames, strict=True)]
            calls.clear()
            ret = dtw.fill_word_timings_batch(
                model, TOK, slot_data, head_mask=torch.ones(1, 1, dtype=torch.bool), max_tokens=max_tokens
            )
            case = {
                "slots": [[t, nf] for t, nf in zip(texts, frames, strict=True)],
                "n_audio_ctx": n_audio,
                "max_tokens": max_tokens,
                "ret": ret,
                "prep": calls[0] if calls else None,
            }
            if model.tokens is not None:
                case["full_tokens"] = model.tokens.tolist()
            out.append(case)
    finally:
        dtw.CrossAttnCapture = orig_capture
        dtw._dtw_prep_batch = orig_prep
    return out


def format_cases():
    """`f"{x:.2f}"` after the 0.02 s quantisation, the only float formatting in the rebuild"""
    vals = [float(v) for v in RNG.uniform(0, 4000, size=300)] + [0.0, 0.005, 0.015, 0.025, 1e-9, 29.99, 1234.565]
    out = []
    for v in vals:
        q = max(0.0, round(v * 50) / 50)
        out.append([v, f"{q:.2f}", f"{v:.2f}"])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    texts = [*TEXTS, *LONG, *trace_texts()]
    tok = {
        "sot_sequence": list(TOK.sot_sequence),
        "no_timestamps": TOK.no_timestamps,
        "eot": TOK.eot,
        "chars": char_table(texts),
    }
    with open(os.path.join(OUT, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tok, f, ensure_ascii=False, indent=0, sort_keys=True)

    cases = {
        "kernel": kernel_cases(),
        "neg_matrix": neg_matrix_cases(),
        "bands": band_cases(),
        "parse": parse_cases(),
        "chunk": chunk_cases(),
        "rebuild": rebuild_cases(),
        "format": format_cases(),
        "batch": batch_cases(),
        "py_space": [cp for cp in range(0x110000) if chr(cp).isspace()],
    }
    with open(os.path.join(OUT, "cases.json"), "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, allow_nan=False)
    print(f"wrote {OUT}: {len(tok['chars'])} chars, " + ", ".join(f"{k}={len(v)}" for k, v in cases.items()))


if __name__ == "__main__":
    main()
