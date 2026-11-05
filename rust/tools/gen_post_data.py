"""Write expected values for the native post-processing and speaker-remap tests

Calls the frozen Python reference on the CPU and writes JSON under
`crates/whisperdrz-native/tests/data/`:

- `pyrandom.json`: CPython `random` draws (`random`, `uniform`, `getrandbits`,
  `randint`) for several seeds
- `torch_cpu.json`: torch CPU f32 bits for `F.normalize`, `pow(2).sum(-1)` and
  `cdist` (25 rows or fewer), plus `cluster_speakers` labels on synthetic sets
- `text_cases.json`: every ported text function on hand-written edge cases and a
  seeded fuzz corpus, with the return value or the exception raised

Usage (CPU only):
    CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python -I gen_post_data.py SRC OUT_DIR
"""

import json
import os
import random
import re
import sys


def bits(t):
    import numpy as np

    return t.contiguous().view(-1).numpy().view(np.uint32).tolist()


def gen_random():
    out = []
    for seed in [0, 1, 12345, 2**40 + 7]:
        r = random.Random(seed)
        calls = []
        for i in range(400):
            op = i % 5
            if op == 0:
                calls.append(["random", [], r.random()])
            elif op == 1:
                calls.append(["uniform", [3.0, 10.0], r.uniform(3.0, 10.0)])
            elif op == 2:
                k = [1, 5, 31, 32, 33, 48, 64][i % 7]
                calls.append(["getrandbits", [k], r.getrandbits(k)])
            elif op == 3:
                b = [0, 1, 2, 7, 153425, 2**31 - 1, 2**31, 2**32, 2**33 + 5][i % 9]
                calls.append(["randint", [0, b], r.randint(0, b)])
            else:
                calls.append(["randint", [-5, 5], r.randint(-5, 5)])
        out.append({"seed": seed, "calls": calls})
    return out


def gen_torch():
    import torch

    from whisperdrz.speaker_clustering import cluster_speakers

    g = torch.Generator().manual_seed(7)
    x = torch.randn(40, 192, generator=g) * 0.07
    xn = torch.nn.functional.normalize(x, p=2, dim=-1)
    out = {
        "x": bits(x),
        "normalize": bits(xn),
        "sumsq": bits(xn.pow(2).sum(-1, True)),
        "cdist25": bits(torch.cdist(xn[:25], xn[:25])),
        "cdist7x3": bits(torch.cdist(xn[:7], xn[30:33])),
        # a column sum past 16 rows exercises the cascade levels
        "colsum40": bits(x.sum(0)),
        "clusters": [],
    }
    # (rows, centers, threshold, min_cluster_size, outliers); outliers are isolated
    # unit vectors, which a min_cluster_size above 1 folds into the nearest large cluster
    cases = [
        (5, 2, 0.6, 1, 0),
        (12, 3, 0.9, 1, 0),
        (25, 4, 1.0, 1, 0),
        (40, 5, 1.0, 1, 0),
        (24, 3, 1.1, 3, 0),
        (60, 6, 1.0, 4, 0),
        (20, 3, 1.0, 2, 3),
        (37, 2, 1.0, 3, 3),
    ]
    for n, k, threshold, min_size, outliers in cases:
        centers = torch.nn.functional.normalize(torch.randn(k, 192, generator=g), dim=-1)
        idx = torch.randint(0, k, (n,), generator=g)
        emb = torch.nn.functional.normalize(centers[idx] + 0.06 * torch.randn(n, 192, generator=g), dim=-1)
        if outliers:
            extra = torch.nn.functional.normalize(torch.randn(outliers, 192, generator=g), dim=-1)
            emb = torch.cat([emb[: n // 2], extra, emb[n // 2 :]])
            n += outliers
        items = [(i, 0, emb[i]) for i in range(n)]
        m = cluster_speakers(items, threshold=threshold, min_cluster_size=min_size)
        out["clusters"].append(
            {
                "emb": bits(emb),
                "threshold": threshold,
                "min_cluster_size": min_size,
                "labels": [m[(i, 0)] for i in range(n)],
            }
        )
    return out


# pieces the fuzz corpus is built from: markers, tags, words, separators and the
# odd characters the ports treat specially
PIECES = [
    "<|0.00|>", "<|1.02|>", "<|1.00|>", "<|2.50|>", "<|0.98|>", "<|12.345|>", "<|30.00|>", "<|١.٥٠|>",
    "<a.96|1.10|>", "<a.12|1.40|>", "<a.92|0.5|>", "<|1.2.3|>", "<|.5|>",
    "[0]", "[1]", "[2]", "[01]", "[c]", "[]", "[CHOP]", "[\"", "[", "]", "!", " 1", " 0",
    "[breath]", "[laugh]", "[pause]", "[music]", "[/music]", "[stopmusic]", "[singing]", "[SİNGİNG]",
    "[singstop]", "[/sing]", "[sing]", "[ſinging]", "[mispeaK]", "[whisper]", "[Whisper]", "[dc]",
    "word", "Hello", "it's", "é", "²", "Councill", "xCouncill", "_", "naïve",
    " ", " ", "  ", "\t", "\n", "\n", "\n\n", " ", "\x1c", " ", "　",
]


def fuzz_strings(n, seed):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        out.append("".join(r.choice(PIECES) for _ in range(r.randint(1, 40))))
    return out


HAND = [
    "",
    "\n",
    "[0] <|0.00|>Hello<|0.50|> <|0.52|>world.<|1.00|>",
    "[0] <|26.68|>I'm<|26.78|> <|26.08|>doing.<|27.08|>\n[1] <|1.00|>a<|0.50|> <|0.40|>b<|0.30|>",
    "[0] <|0.125|>a<|0.375|> <|2.675|>b<|1.005|>",
    "a[0] b\n[1] c[c] d[2] e",
    "é[0]x[1]\n[1] y",
    "xCouncill<|1.0|> Councill<|2.0|> _Councill ²Councill é Councill",
    "[singing] la [/singing] [stopsing] [sing][/sing] [SİNGİNG] x [singingstop]",
    "<|5.14|>[breath]<|5.24|> <|5.26|>[breath]<|5.30|> <|5.40|>[breath]<|5.50|>",
    "[breath] [breath]\t[breath]\n[breath] [laugh] [laugh]",
    "<|1.00|>word<|1.02|> <|1.50|>word<|1.40|>\n<|1.00|>\n[0] <|2.00|>word<|2.02|>",
    "<|1.00|>[breath]<|1.01|> <|1.00|>[pause]<|1.50|> <|1.00|>[pause]<|2.00|>",
    "[0] <|0.08|><a.96|0.14|>word<a.96|0.20|><|0.20|> <|0.30|><a.40|0.31|>x<a.40|0.40|><|0.42|>",
    "\n[0] <|0.00|>a<|0.10|> <|0.20|>b<|0.30|> <|0.40|>c\n[1] <|0.50|>",
    "[0] <|1.00|>a\n<|2.00|>",
    "[c] <|1.00|>cheer<|2.00|>\n[0] <|3.00|>x<|3.00|>\n[1] <|4.00|>y<|5.00|>",
    # the Kelvin sign lowercases to `k`, so this opens a real `mispeak` pair
    "[mispea\u212a] x [/mispeak]\n[mispea\u212a] y\n[MISPEAK] z [mispeakstop]\n[sİng] w [/sing]",
]

# offsets and steps the timing functions see, including exact .xx5 ties
OFFSETS = [0.0, 0, 0.005, 0.015, 0.125, 1.42, 29.99, 120.0, -0.5]


def call(fn, *args):
    try:
        return {"ret": fn(*args)}
    except Exception as e:  # noqa: BLE001 -- the raise is the expected value
        return {"raise": type(e).__name__}


def clean_chunk(sofar, score):
    """Verbatim copy of the cleanup chain in Transcriber._finalize_slot_after_alignment"""
    from whisperdrz.clean import (
        consolidate_adjacent_tags,
        fix_nonmonotonic_timestamps,
        fix_speaker_ids,
        merge_consecutive_speakers,
        reconcile_decoder_and_alignment_timings,
        remove_invalid_timing_pairs,
        remove_orphan_paired_tags,
        remove_short_nonspeech,
        remove_short_pauses,
        remove_speaker_tags_not_after_newline,
    )

    cleaned = sofar
    cleaned = reconcile_decoder_and_alignment_timings(cleaned, confidence_threshold=score)
    cleaned = consolidate_adjacent_tags(cleaned)
    cleaned = remove_invalid_timing_pairs(cleaned)
    cleaned = remove_speaker_tags_not_after_newline(cleaned)
    cleaned = merge_consecutive_speakers(cleaned)
    cleaned = fix_nonmonotonic_timestamps(cleaned)
    cleaned = re.sub(r'\s*\["\s*(?=\n|$)', "", cleaned)
    cleaned = re.sub(r"\s+[01]\s*(?=\n|$)", "", cleaned)
    cleaned = re.sub(r"(?<=\n)]|^]", "", cleaned)
    cleaned = re.sub(r"\[(?=\n|$)", "", cleaned)
    cleaned = re.sub(r"(?<![a-zA-Z])!(?![a-zA-Z])", "", cleaned)
    cleaned = re.sub(r"\bCouncill(?:<\|[\d.]+\|>)?", "", cleaned)
    cleaned = remove_orphan_paired_tags(cleaned)
    cleaned = re.sub(
        r"^\[(?:\d+|c)\]\s*(?:<\|[\d.]+\|>\s*)*(?:\[[a-zA-Z]+\]\s*(?:<\|[\d.]+\|>\s*)*)+\s*$",
        "",
        cleaned,
        flags=re.MULTILINE,
    )
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    cleaned = remove_short_nonspeech(cleaned, min_duration_s=0.02)
    cleaned = remove_short_pauses(cleaned, min_pause_ms=800)
    cleaned = re.sub(r"(\[\d+\] )<\|[\d.]+\|> (<\|[\d.]+\|>)", r"\1\2", cleaned)
    speaker_order = []
    for m in re.finditer(r"\[(\d+)\]", cleaned):
        sid = int(m.group(1))
        if sid not in speaker_order:
            speaker_order.append(sid)
    return [fix_speaker_ids(cleaned), speaker_order]


def gen_text():
    from whisperdrz import clean
    from whisperdrz.clean import timings
    from whisperdrz.inference_utils import InferenceState, finalize_transcription
    from whisperdrz.speaker_clustering import normalize_speaker_ids
    from whisperdrz.transcriber import _strip_to_last_closing, parse_to_segments

    texts = HAND + fuzz_strings(200, 11)
    cases = []

    def add(fn, args, res):
        cases.append({"fn": fn, "args": args, **res})

    unary = {
        "cut_after_last_timed_word": clean.cut_after_last_timed_word,
        "fix_nonmonotonic_timestamps": clean.fix_nonmonotonic_timestamps,
        "fix_speaker_ids": clean.fix_speaker_ids,
        "merge_consecutive_speakers": clean.merge_consecutive_speakers,
        "remove_speaker_tags_not_after_newline": clean.remove_speaker_tags_not_after_newline,
        "consolidate_adjacent_tags": clean.consolidate_adjacent_tags,
        "remove_orphan_paired_tags": clean.remove_orphan_paired_tags,
        "remove_invalid_timing_pairs": clean.remove_invalid_timing_pairs,
        "remove_short_nonspeech": clean.remove_short_nonspeech,
        "remove_short_pauses": clean.remove_short_pauses,
        "normalize_speaker_ids": normalize_speaker_ids,
        "strip_to_last_closing": _strip_to_last_closing,
        "parse_to_segments": parse_to_segments,
    }
    for i, t in enumerate(texts):
        for name, fn in unary.items():
            add(name, [t], call(fn, t))
        add("reconcile_decoder_and_alignment_timings", [t, 40], call(clean.reconcile_decoder_and_alignment_timings, t, 40))
        add("clean_chunk", [t, 40], call(clean_chunk, t, 40))
        off = OFFSETS[i % len(OFFSETS)]
        add("quantize_timestamps", [t, off], call(clean.quantize_timestamps, t, off))
        add("offset_timestamps", [t, off], call(clean.offset_timestamps, t, off))
        for cutearly in (False, True):
            st = InferenceState()
            st.offset = off
            st.cutearly = cutearly
            for apply in (False, True):
                add(
                    "finalize_transcription",
                    [{"offset": off, "cutearly": cutearly}, t, apply],
                    call(finalize_transcription, st, t, apply),
                )

    r = random.Random(5)
    floats = [0.125, 0.375, 2.675, 1.005, 0.005, 0.015, 1e-9, 29.995, 30.005, 120.125]
    floats += [round(r.uniform(0, 200), r.randint(2, 5)) for _ in range(300)]
    floats += [r.uniform(0, 200) for _ in range(300)]
    for x in floats:
        add("fmt_2f", [x], {"ret": f"{x:.2f}"})
        add("round2", [x], {"ret": round(x, 2)})
        add("round_to", [x, 0.02], {"ret": timings.round_to(x, 0.02)})
        add("round_to", [-x / 1000, 0.02], {"ret": timings.round_to(-x / 1000, 0.02)})
    return cases


def main():
    src, out_dir = sys.argv[1], sys.argv[2]
    sys.path.insert(0, os.path.abspath(src))
    import whisperdrz

    assert os.path.abspath(whisperdrz.__file__).startswith(os.path.abspath(src)), whisperdrz.__file__
    os.makedirs(out_dir, exist_ok=True)
    for name, gen in [("pyrandom", gen_random), ("torch_cpu", gen_torch), ("text_cases", gen_text)]:
        with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(gen(), fh, ensure_ascii=False)


if __name__ == "__main__":
    main()
