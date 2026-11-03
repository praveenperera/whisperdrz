"""Replays the speaker remap with the native ECAPA2 embeddings in place of torch's

For each recorded `remap_speakers_per_line` call in a fixture set (short3 by default),
runs the frozen reference source's own `remap_speakers_per_line` on the recorded text
and audio, with the recorded `random.uniform`/`randint` draws replayed in order and
`_embed_audio_segment` answering from an embeddings file instead of the model:

  recorded   the embeddings the trace recorded (a control: must reproduce the trace)
  fp32       the native runtime in FP32, from `ours.json`
  reference  the native runtime with TF32 convolutions, from `ours.json`
  truth      torch in FP32 with TF32 off, from the `manifest.json` beside `ours.json`

and compares each output text with the recorded one, line by line. Also reports how
far each variant moves the pairwise distances AHC clusters on, against the closest
recorded distance to the threshold. CPU only; `ours.json` comes from the `gpu_ecapa2`
test (`scripts/gpu-test.sh`)

  CUDA_VISIBLE_DEVICES="" ~/code/whisperdrz/.venv/bin/python -I tools/ecapa2_remap_check.py \\
      ~/.cache/whisperdrz/ecapa2/ref/ours.json
"""

import argparse
import json
import os
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ecapa2  # noqa: E402

SPEAKER = re.compile(r"^\[(\w+)\]")


def events(set_name, stem):
    with open(os.path.join(ecapa2.FIXTURES, set_name, stem, "events.jsonl")) as f:
        return [json.loads(line) for line in f]


def speakers(text):
    """The speaker tag of each tagged line"""
    return [m.group(1) for line in text.split("\n") if (m := SPEAKER.match(line.strip()))]


class Model:
    """Stands in for the TorchScript model: the remap only asks it for its device"""

    def parameters(self):
        import torch

        yield torch.zeros(1)


def replay(T, torch, ev_list, audio, embeddings):
    """Runs the recorded remap call with `embeddings` (one per embed call, in order)"""
    remap = next(e for e in ev_list if e["ev"] == "remap_speakers_per_line")
    draws = [e for e in ev_list if e["ev"].startswith("random.")]
    embeds = [e for e in ev_list if e["ev"] == "_embed_audio_segment"]
    assert len(embeds) == len(embeddings), (len(embeds), len(embeddings))
    start, n = remap["args"]["audio"]["__audio"]
    line_audio = audio[start : start + n]

    queue = list(draws)

    def draw(name):
        def f(*args):
            ev = queue.pop(0)
            assert ev["ev"] == f"random.{name}" and list(args) == ev["args"], (name, args, ev)
            return ev["ret"]

        return f

    calls = list(zip(embeds, embeddings))

    def embed(audio_tensor, model, device):
        ev, e = calls.pop(0)
        s, k = ev["args"]["audio_tensor"]["__audio"]
        want = torch.from_numpy(audio[s : s + k])
        assert audio_tensor.shape == want.shape and torch.equal(audio_tensor, want), "embed input differs"
        return torch.tensor(e, dtype=torch.float32)

    saved = (random.uniform, random.randint, T._embed_audio_segment)
    random.uniform, random.randint, T._embed_audio_segment = draw("uniform"), draw("randint"), embed
    try:
        kwargs = {k: v for k, v in remap["args"].items() if k in ("threshold", "min_duration")}
        out = T.remap_speakers_per_line(remap["args"]["text"], torch.from_numpy(line_audio), Model(), **kwargs)
    finally:
        random.uniform, random.randint, T._embed_audio_segment = saved
    assert not queue and not calls, "unused recorded draws or embeddings"
    return out, remap["ret"], kwargs.get("threshold", 1.1)


def distances(v):
    v = np.asarray(v, dtype=np.float64)
    d = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=-1)
    return d[np.triu_indices(len(v), 1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ours", help="ours.json from the gpu_ecapa2 test")
    p.add_argument("--set", default="short3")
    args = p.parse_args()

    summary = json.load(open(os.path.join(ecapa2.FIXTURES, args.set, "summary.json")))
    sys.path.insert(0, summary["src"])
    import torch
    from whisperdrz import transcriber as T

    ours_path = os.path.expanduser(args.ours)
    ours = json.load(open(ours_path))
    manifest = json.load(open(os.path.join(os.path.dirname(ours_path), "manifest.json")))
    truth = {s["name"]: s["truth"] for s in manifest["segments"]}
    failed = []
    print(f"{'stem':<8} {'variant':<10} {'lines':>5} {'changed':>7} {'same text':>9} {'max |dd|':>9} {'min |d-thr|':>11}")
    for f in summary["files"]:
        stem = f["stem"]
        ev_list = events(args.set, stem)
        audio = ecapa2.file_audio(args.set, stem)
        embeds = [e for e in ev_list if e["ev"] == "_embed_audio_segment"]
        recorded = [e["ret"]["data"] for e in embeds]
        variants = {"recorded": recorded}
        for mode in ("fp32", "reference"):
            variants[mode] = [ours[mode][f"{args.set}-{stem}-{e['seq']}"]["embedding"] for e in embeds]
        variants["truth"] = [truth[f"{args.set}-{stem}-{e['seq']}"] for e in embeds]

        d_rec = distances(recorded)
        for name, emb in variants.items():
            out, want, threshold = replay(T, torch, ev_list, audio, emb)
            got_s, want_s = speakers(out), speakers(want)
            changed = sum(a != b for a, b in zip(got_s, want_s)) + abs(len(got_s) - len(want_s))
            dd = np.abs(distances(emb) - d_rec).max() if len(emb) > 1 else 0.0
            gap = np.abs(d_rec - threshold).min() if len(emb) > 1 else float("nan")
            print(f"{stem:<8} {name:<10} {len(want_s):>5} {changed:>7} {str(out == want):>9} {dd:>9.2e} {gap:>11.3e}")
            if out != want:
                failed.append(f"{stem} {name}")

    if failed:
        print(f"remap output differs: {failed}")
        return 1
    print("remap decisions unchanged for every variant")
    return 0


if __name__ == "__main__":
    sys.exit(main())
