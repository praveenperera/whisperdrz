"""Noise-band output gate for the native runtime

usage: gate.py {vox24|e21} NEW_TEXTS_JSON

NEW_TEXTS_JSON maps each wav basename to {"text", "segments"} like `<workload>.ref.json`.
Byte identity with the Python reference isn't the bar: rerunning the reference with one
change that is exact in real arithmetic (f64 mel, memory-efficient encoder SDPA,
cross-attention without split-KV) changed words or speakers on 10-17 of 24 vox24 files
and on e21, so the gate passes a run whose distance from the reference and whose DER
against the RTTMs stay inside the band those reruns measured. It still prints whether
the run is byte-identical, and the first divergence per file, to localize bugs

Run with the whisperdrz venv (pyannote.metrics, jiwer, meeteval)
"""

import contextlib
import hashlib
import io
import json
import math
import os
import re
import sys
import warnings

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from whisperdrz.evals import calculate_cpwer, calculate_wer  # noqa: E402

# pyannote warns that the scoring extent comes from the inputs, which is what DER wants
warnings.filterwarnings("ignore", message=".uem. was approximated")
RTTM = "/home/praveen/code/speakrs/fixtures/datasets/{ds}/rttm/{uri}.rttm"

# limits from the perturbation reruns (decision log 2026-11-03 13:38 CT). vox24: reference
# DER 21.98, arms 20.73-21.75, word edits up to 0.85% and cpWER-style edits up to 5.37%
# (mel64); e21: DER 27.47, arms 27.42-27.47, edits up to 0.042% and 0.146%. Each limit is
# the worst arm with headroom, since three arms sample the noise thinly
LIMITS = {
    "vox24": {"ds": "voxconverse-dev", "der": 21.98 + 1.25, "word_edit": 1.0, "cp_edit": 6.0},
    "e21": {"ds": "earnings21", "der": 27.47 + 0.5, "word_edit": 0.1, "cp_edit": 0.3},
}
TIMESTAMP = re.compile(r"<\|[0-9.]+\|>")


def gate_hash(out):
    texts = [out[k]["text"] for k in out]
    segments = [out[k]["segments"] for k in out]
    blob = json.dumps({"texts": texts, "segments": segments}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def rttm(uri, ds):
    ann = Annotation(uri=uri)
    for line in open(RTTM.format(ds=ds, uri=uri)):
        f = line.split()
        if f and f[0] == "SPEAKER":
            start, dur = float(f[3]), float(f[4])
            ann[Segment(start, start + dur)] = f[7]
    return ann


def hypothesis(uri, segments):
    ann = Annotation(uri=uri)
    for s in segments:
        if s["end"] > s["start"]:
            ann[Segment(float(s["start"]), float(s["end"])), len(ann)] = str(s["speaker"])
    return ann


def check_shape(ref, new):
    """Fails loudly on output a numeric comparison would let through"""
    problems = []
    if list(ref) != list(new):
        problems.append(f"files {sorted(set(ref) ^ set(new))} missing or extra, or out of order")
    for k in ref:
        if k not in new:
            continue
        text, segs = new[k].get("text"), new[k].get("segments")
        if not isinstance(text, str) or not isinstance(segs, list):
            problems.append(f"{k}: text or segments missing")
            continue
        if ref[k]["text"] and not text.strip():
            problems.append(f"{k}: empty text")
        if ref[k]["segments"] and not segs:
            problems.append(f"{k}: no segments")
        for s in segs:
            if not all(isinstance(s.get(f), (int, float)) and math.isfinite(s[f]) for f in ("start", "end")):
                problems.append(f"{k}: non-finite segment time {s}")
                break
    return problems


def first_divergence(a, b):
    i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return i, a[max(0, i - 40) : i + 40], b[max(0, i - 40) : i + 40]


def main():
    workload, new_path = sys.argv[1], sys.argv[2]
    limits = LIMITS[workload]
    ref = json.load(open(os.path.join(HERE, f"{workload}.ref.json")))
    new = json.load(open(new_path))

    problems = check_shape(ref, new)
    for p in problems:
        print(f"SHAPE {p}")
    if problems:
        print(f"GATE FAIL {workload}: malformed output")
        return 1

    identical = gate_hash(new) == gate_hash(ref)
    metric_ref = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    metric_new = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    edits = words = cp_edits = 0
    differ = 0
    for k in ref:
        uri = os.path.splitext(k)[0]
        truth = rttm(uri, limits["ds"])
        metric_ref(truth, hypothesis(uri, ref[k]["segments"]))
        metric_new(truth, hypothesis(uri, new[k]["segments"]))
        if new[k] == ref[k]:
            continue

        differ += 1
        a, b = TIMESTAMP.sub("", ref[k]["text"]), TIMESTAMP.sub("", new[k]["text"])
        w = calculate_wer(a, b)
        # meeteval prints a sort note per call
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            c = calculate_cpwer(a, b)
        n = w["insertions"] + w["deletions"] + w["substitutions"]
        edits += n
        cp_edits += c["errors"]
        if a == b:
            print(f"  {k}: timestamps or segments only")
        else:
            i, x, y = first_divergence(a, b)
            print(f"  {k}: {n} word edits, first at char {i}: ref {x!r} | new {y!r}")
    words = sum(calculate_wer(TIMESTAMP.sub("", ref[k]["text"]), "")["reference_words"] for k in ref)

    der_ref, der_new = 100 * abs(metric_ref), 100 * abs(metric_new)
    word_edit, cp_edit = 100 * edits / words, 100 * cp_edits / words
    checks = [
        ("der", der_new, limits["der"]),
        ("word_edit", word_edit, limits["word_edit"]),
        ("cp_edit", cp_edit, limits["cp_edit"]),
    ]
    print(f"{workload}: byte-identical={identical} files_differ={differ}/{len(ref)} DER ref {der_ref:.2f} new {der_new:.2f}")
    print(f"{workload}: word edits {edits}/{words} = {word_edit:.3f}%, cp edits {cp_edits} = {cp_edit:.3f}%")
    failed = [f"{name} {value:.3f} > {limit:.3f}" for name, value, limit in checks if value > limit]
    print(f"GATE {'FAIL' if failed else 'PASS'} {workload}" + (f": {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
