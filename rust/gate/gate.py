"""Noise-band output gate for the native runtime

usage: gate.py {vox24|e21} NEW_TEXTS_JSON

NEW_TEXTS_JSON maps each wav basename to {"text", "segments"} like `<workload>.ref.json`.
Byte identity with the Python reference isn't the bar: rerunning the reference with one
change that is exact in real arithmetic (f64 mel, memory-efficient encoder SDPA,
cross-attention without split-KV) changed words or speakers on 10-17 of 24 vox24 files
and on e21, so the gate passes a run whose distance from the reference and whose DER
against the RTTMs stay inside the band those reruns measured, and whose word timings
(coverage, mean boundary error, share moved over 0.2 s) stay inside that band too. It still prints whether
the run is byte-identical, and the first divergence per file, to localize bugs

Run with the whisperdrz venv (pyannote.metrics, jiwer, meeteval)
"""

import contextlib
import difflib
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
    # timing limits are 1.5x the worst of the mel64, enc-eff and cross-ns1 reference
    # reruns: mean boundary error vox24 0.209 s, e21 0.178 s; boundaries moved over
    # 0.2 s vox24 2.93%, e21 2.96%. The timed-word fraction of every rerun stayed within
    # 0.13 points of the reference. cp limits are 1.5x the worst cp rate of all five
    # reruns (those three plus dec-f32 and gemm-lt): vox24 mel64 1465 of 27276 words,
    # e21 gemm-lt 63 of 9613. A BLAS-library switch alone took the reference past the
    # earlier e21 cap of 0.3%. The 1.5x is an engineering margin, not a confidence bound
    "vox24": {"ds": "voxconverse-dev", "der": 21.98 + 1.25, "word_edit": 1.0,
              "cp_edit": 1.5 * 100 * 1465 / 27276,
              "timed_drop": 0.5, "t_mean": 0.313, "t_over": 4.39},
    "e21": {"ds": "earnings21", "der": 27.47 + 0.5, "word_edit": 0.1,
            "cp_edit": 1.5 * 100 * 63 / 9613,
            "timed_drop": 0.5, "t_mean": 0.266, "t_over": 4.44},
}
TIMESTAMP = re.compile(r"<\|[0-9.]+\|>")
# a word with the timestamps written right against it, `<|s|>word<|e|>`
TIMED_WORD = re.compile(r"(?:<\|([0-9.]+)\|>)?([^\s<]+)(?:<\|([0-9.]+)\|>)?")
SPEAKER_TAG = re.compile(r"\[\d+\]")
# boundaries moving more than this count toward `t_over`
TIMING_SLACK = 0.2


def timed_words(text):
    """(word, start or None, end or None) for each word, speaker tags dropped"""
    out = []
    for s, w, e in TIMED_WORD.findall(text):
        if not SPEAKER_TAG.fullmatch(w):
            out.append((w, float(s) if s else None, float(e) if e else None))
    return out


def timing(ref, new):
    """Word-timing coverage and error: the text metrics strip timestamps and DER reads
    only segments, so without this a run that drops or shifts word timings passes.
    Words are aligned over the whole timestamp-free word sequence, and a boundary's error
    is taken where both runs time the same aligned word"""
    errors, words_ref, words_new, timed_ref, timed_new = [], 0, 0, 0, 0
    for k in ref:
        a, b = timed_words(ref[k]["text"]), timed_words(new[k]["text"])
        words_ref += len(a)
        words_new += len(b)
        timed_ref += sum(w[1] is not None for w in a)
        timed_new += sum(w[1] is not None for w in b)
        sm = difflib.SequenceMatcher(None, [w[0] for w in a], [w[0] for w in b], autojunk=False)
        for blk in sm.get_matching_blocks():
            for i in range(blk.size):
                x, y = a[blk.a + i], b[blk.b + i]
                errors += [abs(x[j] - y[j]) for j in (1, 2) if x[j] is not None and y[j] is not None]
    frac_ref = 100 * timed_ref / max(words_ref, 1)
    frac_new = 100 * timed_new / max(words_new, 1)
    if not errors:
        return frac_ref, frac_new, math.inf, 100.0
    mean = sum(errors) / len(errors)
    over = 100 * sum(e > TIMING_SLACK for e in errors) / len(errors)
    return frac_ref, frac_new, mean, over


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
    frac_ref, frac_new, t_mean, t_over = timing(ref, new)
    checks = [
        ("der", der_new, limits["der"]),
        ("word_edit", word_edit, limits["word_edit"]),
        ("cp_edit", cp_edit, limits["cp_edit"]),
        ("timed_drop", frac_ref - frac_new, limits["timed_drop"]),
        ("t_mean", t_mean, limits["t_mean"]),
        ("t_over", t_over, limits["t_over"]),
    ]
    print(f"{workload}: byte-identical={identical} files_differ={differ}/{len(ref)} DER ref {der_ref:.2f} new {der_new:.2f}")
    print(f"{workload}: word edits {edits}/{words} = {word_edit:.3f}%, cp edits {cp_edits} = {cp_edit:.3f}%")
    print(f"{workload}: timed words ref {frac_ref:.2f}% new {frac_new:.2f}%, boundary error mean {t_mean:.3f} s, {t_over:.2f}% moved > {TIMING_SLACK} s")
    failed = [f"{name} {value:.3f} > {limit:.3f}" for name, value, limit in checks if value > limit]
    print(f"GATE {'FAIL' if failed else 'PASS'} {workload}" + (f": {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
