"""Write the CPU reference data the Rust tokenizer and decoder tables are tested against

Runs the frozen Python on the CPU only (no model, no GPU): builds the large-v3
tokenizer, runs `inference_utils.init_for_model` with a stand-in model of vocab
51866, builds `DecoderState(tokenizer, device="cpu")` and dumps

  decode_tables.json  every list and bool mask of DecoderState and inference_utils,
                      per-language base masks, per-character string properties
  tokenizer.json      encode/decode cases: edge cases, seeded random text, random
                      token sequences, and sha256 digests for the trace texts
  decode_cases.json   hand-built slot states for the decode_batch and
                      update_state_after_decode branches the traces never reach, with
                      python's results; decode_batch runs them as one batch and one row
                      at a time and asserts both agree

Small index sets are stored as sorted `[start, end]` runs, large ones as their size
and the sha256 of the comma-joined ids (sorted, or in python's order where python's
order is deterministic); `--full` writes every set out in full for diffing. Usage:

  CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 python -I \\
      rust/tools/dump_decode_reference.py --src SNAPSHOT/src \\
      --fixtures ~/.cache/whisperdrz/fixtures/ref-a783291 \\
      --out rust/crates/whisperdrz-native/tests/data
"""

import argparse
import hashlib
import json
import os
import random
import sys
import unicodedata


def runs(ids):
    """Sorted unique ids as [start, end] inclusive runs"""
    out = []
    for i in sorted(set(int(x) for x in ids)):
        if out and out[-1][1] == i - 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def mask_ids(t):
    return [int(i) for i in t.nonzero().flatten().tolist()]


def ids_digest(ids):
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()


FULL = False
# sets with more runs than this are committed as a digest
MAX_RUNS = 64


def id_set(ids):
    """An unordered set: runs when small, else size and digest of the sorted ids"""
    ids = sorted(set(int(x) for x in ids))
    r = runs(ids)
    if FULL or len(r) <= MAX_RUNS:
        return {"runs": r}
    return {"n": len(ids), "sha256": ids_digest(ids)}


def id_list(ids, ordered):
    """A list: in python's order when `ordered`, else sorted (duplicates kept)"""
    ids = [int(x) for x in ids] if ordered else sorted(int(x) for x in ids)
    if FULL or len(ids) <= 2 * MAX_RUNS:
        return {"ids": ids, "ordered": ordered}
    return {"n": len(ids), "sha256": ids_digest(ids), "ordered": ordered}


def trace_texts(fixtures):
    """Final texts and every distinct `cur` the decoder saw, in first-seen order"""
    seen = {}
    for set_name in sorted(os.listdir(fixtures)):
        summary = os.path.join(fixtures, set_name, "summary.json")
        if not os.path.exists(summary):
            continue
        for f in json.load(open(summary))["files"]:
            stem = f["stem"]
            seen.setdefault(json.load(open(os.path.join(fixtures, set_name, stem, "result.json")))["text"], None)
            for line in open(os.path.join(fixtures, set_name, stem, "events.jsonl"), encoding="utf-8"):
                ev = json.loads(line)
                if ev["ev"] != "update_state_after_decode":
                    continue
                seen.setdefault(ev["after"]["state"]["__state"]["cur"], None)
    return list(seen)


def edge_cases():
    return [
        "",
        " ",
        "  ",
        "\n",
        "\n\n",
        " \n ",
        "\t\t x",
        "Hello world",
        " Hello  world  ",
        "[0] <|0.12|>Hi<|0.24|> <|0.24|>there<|0.50|>\n[1] <|0.60|>[laugh]<|0.90|>",
        "<|endoftext|><|startoftranscript|><|en|><|transcribe|><|notimestamps|><|30.00|>",
        "<|0.00|> <|29.98|>",
        "<|nottoken|> <|0.123|> <|1.0|>",
        "it's we're they've I'm you'll he'd 'S 'T",
        "1234567890 12,345.67 3.14159 1e10 0x1F 007",
        "²³¹ ⁴ ₂ ① ⑴ ⒈ ⓪ ❶ ½ Ⅻ ٣٤ ३ ๓",
        "naïve café résumé Zoë Ångström",
        "日本語のテキスト。中文文本，한국어 텍스트",
        "Привет, мир! Γειά σου κόσμε. שלום עולם. مرحبا بالعالم",
        "हिन्दी पाठ தமிழ் ภาษาไทย",
        "emoji 😀🎉👍🏽 👨‍👩‍👧 flags 🇺🇸",
        "zero​width‍joiner﻿bom",
        "\x00\x01\x1c\x1f\x7f\x85\xa0 　",
        "tabs\tand\rcarriage\x0breturns\x0c",
        "**bold** __under__ ~~strike~~ `code` #hash",
        "a--b..c!!d??e,,f''g...h…i–j—k“l”m‘n’o´p",
        "(paren) [bracket] {brace} <angle> |pipe|",
        "    leading and trailing    ",
        "word" * 50,
        " " * 33 + "x",
        "̀́ combining é",
    ]


def random_texts(tok, n=400, seed=0):
    """Seeded mixes of vocab pieces, Unicode letters, digits and whitespace"""
    rng = random.Random(seed)
    pieces = [tok.decode([i]) for i in range(tok.encoding.n_vocab)]
    alphabet = (
        list("abcXYZ019 \n\t.,!?'-[]<>|*")
        + [chr(c) for c in range(0x00C0, 0x0250, 7)]
        + [chr(c) for c in range(0x0400, 0x0500, 11)]
        + [chr(c) for c in range(0x3040, 0x30FF, 13)]
        + [chr(c) for c in range(0x4E00, 0x4F00, 17)]
        + ["̀", "​", "　", "\xa0", "😀", "²", "٣"]
    )
    out = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(1, 12)):
            r = rng.random()
            if r < 0.5:
                parts.append(rng.choice(pieces))
            elif r < 0.9:
                parts.append("".join(rng.choice(alphabet) for _ in range(rng.randint(1, 6))))
            else:
                parts.append(f"<|{rng.randint(0, 1500) * 0.02:.2f}|>")
        out.append("".join(parts))
    return out


V = 51866
# fields of the synthetic states, as the tracer's snapshots name them
STATE_FIELDS = [
    "language",
    "status",
    "offset",
    "cur",
    "last_timing",
    "token_count",
    "last_token_id",
    "recent_tokens",
    "forced_timing_before_newline",
    "force_timing_every_word",
    "force_timestamp_next",
    "predict_next",
    "stop_reason",
    "_window_max_timing",
    "_timing_count",
    "_line_cache",
]


def synth_logits(spec):
    """The logits row a case names: an integer hash spread over [-4, 4) in steps of 1/128
    (exact in fp16), or a constant `fill`, then `boosts` [[token, value], ...]"""
    import numpy as np

    j = np.arange(V, dtype=np.uint64)
    if "fill" in spec:
        row = np.full(V, spec["fill"], dtype=np.float64)
    else:
        h = (j * np.uint64(2654435761) + np.uint64(spec.get("seed", 0)) * np.uint64(40503)) % np.uint64(2**32)
        row = ((h >> np.uint64(22)).astype(np.float64) - 512.0) / 128.0
    for t, v in spec.get("boosts", []):
        row[t] = v
    return row.astype(np.float16)


def make_state(fields):
    from whisperdrz.inference_utils import InferenceState

    s = InferenceState()
    defaults = {
        "language": "en",
        "status": "started",
        "offset": 0.0,
        "cur": "[0] ",
        "last_timing": None,
        "token_count": 51,
        "last_token_id": 220,
        "recent_tokens": [],
        "forced_timing_before_newline": False,
        "force_timing_every_word": False,
        "force_timestamp_next": False,
        "predict_next": None,
        "stop_reason": None,
        "_window_max_timing": 0.0,
        "_timing_count": 0,
        "_line_cache": "",
    }
    for k, v in {**defaults, **fields}.items():
        if k == "duration":
            continue
        setattr(s, k, v)
    s.meta = {"duration": fields.get("duration", 30.0)}
    return s


def snap(s):
    out = {k: getattr(s, k) for k in STATE_FIELDS}
    out["recent_tokens"] = list(out["recent_tokens"])
    out["meta"] = {"duration": s.meta["duration"]}
    return out


def decode_cases(tok, ds):
    """decode_batch cases: (name, state fields, logits spec, repetition window)"""
    enc = lambda t: tok.encode(t)[0]  # noqa: E731
    t = lambda x: tok.special_tokens[f"<|{x:.2f}|>"]  # noqa: E731
    eot = tok.eot
    rep = tok.encode("'s\n")
    return [
        ("not_started", {"status": "ready"}, {}, None),
        ("max_tokens", {"token_count": 448}, {}, None),
        ("eot_predicted", {"last_token_id": eot}, {}, None),
        # 30 tokens: a 2-token pattern repeated, timings interleaved
        ("repetition", {"token_count": 30, "cur": "[0] <|0.10|>x"}, {}, [rep[0], t(1.0), rep[1]] * 10),
        ("no_repetition", {"token_count": 40, "cur": "[0] <|0.10|>Hi<|0.30|> there"}, {"seed": 3}, list(range(100, 140))),
        ("language", {"token_count": 1, "language": "auto"}, {"boosts": [[tok.special_tokens["<|de|>"], 9.0]]}, None),
        ("transcribe", {"token_count": 2}, {}, None),
        ("force_timestamp", {"force_timestamp_next": True, "cur": "[0] <|1.00|>hello<|1.40|> "}, {"boosts": [[t(1.6), 9.0]]}, None),
        # empty timing range: every token blocked, so the fp16 fill ties and id 0 wins
        ("force_timestamp_empty_range", {"force_timestamp_next": True, "cur": "[0] <|5.00|>word", "duration": 3.0}, {"seed": 5}, None),
        ("defer_newline_empty", {"predict_next": "\n", "cur": "[0] <|0.10|>"}, {}, None),
        ("defer_eot", {"predict_next": "<|endoftext|>", "cur": "[0] <|0.10|>Hi<|0.30|>"}, {}, None),
        ("defer_paren", {"predict_next": "(", "cur": "[0] <|0.10|>Hi<|0.30|> "}, {}, None),
        ("defer_word", {"predict_next": " world", "cur": "[0] <|0.10|>Hi<|0.30|> "}, {}, None),
        ("line_start_empty", {"cur": "[0] <|0.10|>Hi<|0.30|>\n"}, {}, None),
        ("line_start_first", {"cur": "["}, {}, None),
        ("line_start_digit", {"cur": "[0] <|0.10|>Hi<|0.30|>\n["}, {"boosts": [[enc("3"), 9.0], [enc("x"), 12.0]]}, None),
        ("line_start_close", {"cur": "[0] <|0.10|>Hi<|0.30|>\n[1"}, {}, None),
        ("line_start_space", {"cur": "[0] <|0.10|>Hi<|0.30|>\n[1]"}, {}, None),
        ("line_start_timing_same_speaker", {"cur": "[1] <|2.00|>Hi<|2.40|>\n[1] "}, {"boosts": [[t(1.0), 9.0], [t(2.6), 8.0]]}, None),
        ("line_start_timing_other_speaker", {"cur": "[1] <|2.00|>Hi<|2.40|>\n[0] "}, {"boosts": [[t(1.0), 9.0], [t(2.6), 8.0]]}, None),
        ("bracket_prefix", {"cur": "[0] <|0.10|>Hi<|0.30|> <|0.30|>[pau", "last_token_id": enc("pau")}, {"boosts": [[enc("se"), 7.0], [enc(" the"), 9.0]]}, None),
        ("bracket_invalid", {"cur": "[0] <|0.10|>Hi<|0.30|> <|0.30|>[xyz"}, {}, None),
        ("bracket_too_long", {"cur": "[0] <|0.10|>Hi<|0.30|> <|0.30|>[" + "a" * 21}, {}, None),
        ("asterisk_space", {"cur": "[0] <|0.10|>Hi<|0.30|> *foo bar"}, {}, None),
        # eot outscores every unblocked token even after the -65504 fill, so phase 4 resamples it away
        ("eot_resample", {"cur": "[0] hello there", "last_token_id": enc(" there")}, {"fill": -1000.0, "boosts": [[eot, 65000.0], [enc(" now"), -900.0]]}, None),
        ("first_word_timing_rejected", {"cur": "[0] <|0.10|>[laugh]<|0.50|> ", "last_token_id": 220}, {"boosts": [[t(0.6), 9.0], [enc("Hi"), 8.0]]}, None),
        ("space_bracket_split", {"cur": "[0] <|0.10|>Hi", "last_token_id": enc("Hi")}, {"boosts": [[enc(" ["), 9.0]]}, None),
        ("newline_needs_timing", {"cur": "[0] <|0.10|>Hi<|0.30|> there", "last_token_id": enc(" there")}, {"boosts": [[enc("\n"), 9.0]]}, None),
        ("near_end_space", {"cur": "[0] <|2.80|>Hi<|3.00|> there", "duration": 5.0, "last_token_id": enc(" there")}, {"boosts": [[enc(" now"), 9.0]]}, None),
        ("too_many_words", {"cur": "[0] <|0.10|>Hi<|0.30|> a b c d e", "last_token_id": enc(" e")}, {"boosts": [[enc(" f"), 9.0]]}, None),
        ("penalty", {"cur": "[0] <|0.10|>Hi<|0.30|> there", "recent_tokens": [enc(" now")], "last_token_id": enc(" there")}, {"boosts": [[enc(" now"), 5.0], [enc(" then"), 4.5]]}, None),
        ("odd_timing_then_space", {"cur": "[0] <|0.10|>Hi", "last_token_id": enc("Hi")}, {"boosts": [[enc(" there"), 9.0]]}, None),
        ("timing_after_word", {"cur": "[0] <|0.10|>Hi", "last_token_id": enc("Hi")}, {"boosts": [[t(0.2), 9.0]]}, None),
        ("quote_after_quote", {"cur": '[0] <|0.10|>Hi<|0.30|> "', "last_token_id": enc(' "')}, {"boosts": [[enc('"'), 9.0], [enc("x"), 8.0]]}, None),
    ]


def update_cases(tok):
    """update_state_after_decode cases: (name, state fields, newtext, token id)"""
    t = lambda x: tok.special_tokens[f"<|{x:.2f}|>"]  # noqa: E731
    return [
        ("timing_exceeded", {"cur": "[0] <|29.80|>end", "_timing_count": 1, "_line_cache": "[0] <|29.80|>end", "last_timing": 29.8}, "<|29.90|>", t(29.9)),
        ("audio_exhausted", {"cur": "[0] <|18.00|>end<|19.50|>", "_timing_count": 2, "last_timing": 19.5, "_window_max_timing": 19.5, "duration": 20.0}, "\n", 198),
        ("empty_line", {"cur": "[0] <|1.00|>", "_timing_count": 1, "last_timing": 1.0}, "\n", 198),
        ("cutoff_kept_time", {"cur": "[0] <|1.00|>a [cutoff", "_timing_count": 1, "last_timing": 1.0}, "]", 60),
        ("cutoff_stop", {"cur": "[0] <|29.00|>a [cutoff", "_timing_count": 1, "last_timing": 29.0}, "]", 60),
        ("special_text", {"recent_tokens": list(range(10))}, "<|en|>", tok.special_tokens["<|en|>"]),
        ("empty_text", {}, "", tok.transcribe),
        ("plain_word", {"cur": "[0] <|1.00|>a", "_timing_count": 1, "last_timing": 1.0}, " b", tok.encode(" b")[0]),
    ]


def run_cases(tok, ds):
    import torch

    from whisperdrz.decoding import decode_batch, update_state_after_decode

    cases = decode_cases(tok, ds)
    rows = [synth_logits(spec) for _, _, spec, _ in cases]
    all_tokens = torch.zeros((len(cases), 448), dtype=torch.long)
    for i, (_, fields, _, window) in enumerate(cases):
        if window is not None:
            n = fields["token_count"]
            all_tokens[i, max(0, n - 100) : n] = torch.tensor(window)

    def run(indices):
        states = [make_state(cases[i][1]) for i in indices]
        logits = torch.from_numpy(__import__("numpy").stack([rows[i] for i in indices]))
        ret = decode_batch(
            logits,
            states,
            ds,
            last_token_ids=[s.last_token_id for s in states],
            token_counts=[s.token_count for s in states],
            all_tokens_gpu=all_tokens,
            slot_indices=list(indices),
        )
        return [(list(r), snap(s)) for r, s in zip(ret, states, strict=True)]

    batched = run(list(range(len(cases))))
    single = [run([i])[0] for i in range(len(cases))]
    assert batched == single, "decode_batch results depend on batching"

    out_decode = []
    for (name, fields, spec, window), (ret, after) in zip(cases, batched, strict=True):
        out_decode.append(
            {
                "name": name,
                "state": snap(make_state(fields)),
                "logits": spec,
                "window": window,
                "ret": ret,
                "after": after,
            }
        )

    out_update = []
    for name, fields, newtext, token in update_cases(tok):
        s = make_state(fields)
        before = snap(s)
        update_state_after_decode(s, newtext, token)
        out_update.append({"name": name, "state": before, "newtext": newtext, "token": token, "after": snap(s)})
    return {"decode": out_decode, "update": out_update}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--fixtures", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--full", action="store_true", help="write every set in full (for diffing, not for committing)")
    args = p.parse_args()
    global FULL
    FULL = args.full

    sys.path.insert(0, os.path.abspath(args.src))
    import torch

    import whisperdrz.inference_utils as iu
    from whisperdrz.decoding import DecoderState, is_word_char
    from whisperdrz.tokenizer import get_tokenizer
    from whisperdrz.vocab_constraints import _get_token_script

    assert not torch.cuda.is_available(), "run with CUDA_VISIBLE_DEVICES=''"

    # large-v3: n_vocab 51866 -> num_languages = 51866 - 51765 - 1 = 100, as WhisperDRZ.tokenizer computes
    tok = get_tokenizer(True, num_languages=100, language="en")
    assert tok.encoding.n_vocab == 51866

    class Dims:
        n_vocab = 51866

    class Model:
        dims = Dims()
        tokenizer = tok

    iu.init_for_model(Model())
    ds = DecoderState(tok, device="cpu")
    ds.init_gpu_tensors()
    cache = ds._device_tensors["cpu"]

    vocab_digest = hashlib.sha256()
    for t in ds.vocab_list:
        vocab_digest.update(t.encode("utf-8") + b"\xff")

    # python builds these in a deterministic order; the rest go through sets
    ordered = {
        "timings": ds.timings,
        "timing_token_ids": cache["timing_token_ids"].tolist(),
        "space_starting": ds.space_starting,
        "digit_starting": ds.digit_starting,
        "space_tokens_only": ds.space_tokens_only,
        "punctuation": ds.punctuation,
        "bracket_candidates": [t for t, _ in ds._bracket_candidates],
        "angle_tokens_blocked": list(iu.angle_tokens_blocked),
        "initial_tokens": list(iu.initial_tokens),
    }
    for p_char, s in zip(["!", "?", ",", "-", ".", "…", "'"], ds._punct_block_sets, strict=True):
        ordered[f"punct_block_set:{p_char}"] = s
    unordered = {
        "all_punct": ds._all_punct_blocked,
        "lang_tokens": cache["lang_tokens"].tolist(),
        "dont_predict": iu.dont_predict,
        "asterisk_tokens_unblockable": iu.asterisk_tokens_unblockable,
    }
    lists = {k: id_list(v, True) for k, v in ordered.items()}
    lists.update({k: id_list(v, False) for k, v in unordered.items()})

    mask_names = [
        "base_blocked",
        "base_extra_blocked",
        "angle_blocked_mask",
        "asterisk_blocked_mask",
        "block_open_bracket",
        "block_close_bracket",
        "block_open_paren",
        "block_close_paren",
        "block_less_than",
        "block_greater_than",
        "block_angle_consecutive",
        "block_asterisk",
        "space_asterisk_token",
        "block_contains_space",
        "block_contains_close_bracket",
        "block_contains_open_bracket",
        "block_quote",
        "block_timings",
        "block_space_starting",
        "block_digit_starting",
        "block_punct",
        "block_newlines",
        "in_bracket_blocked",
        "in_angle_blocked",
        "in_paren_blocked",
        "after_timing_blocked",
        "lang_enabled",
        "space_starting_enabled",
        "speaker_id_mask",
    ]
    masks = {name: id_set(mask_ids(cache[name])) for name in mask_names}

    langs = ["en", "auto", "ru", "el", "ar", "he", "zh", "ja", "hi", "ur", "th", "hy", "ka", "am", "xx"]
    base_by_lang = {lang: id_set(mask_ids(ds.get_base_blocked_cpu(lang))) for lang in langs}

    bracket_allowed = {k: v for k, v in sorted(ds._cpu_masks["bracket_allowed"].items())}

    chars = sorted({c for t in ds.vocab_list for c in t})
    props = {
        "isalnum": id_set(ord(c) for c in chars if c.isalnum()),
        "isdigit": id_set(ord(c) for c in chars if c.isdigit()),
        "isspace": id_set(ord(c) for c in chars if c.isspace()),
        "isprintable": id_set(ord(c) for c in chars if c.isprintable()),
        "is_word_char": id_set(ord(c) for c in chars if is_word_char(c)),
    }
    scripts = {}
    for c in chars:
        scripts.setdefault(str(_get_token_script(c)), []).append(ord(c))
    props["script"] = {k: id_set(v) for k, v in sorted(scripts.items())}

    tables = {
        "n_vocab": ds.vocab_size,
        "chars": id_set(ord(c) for c in chars),
        "vocab_sha256": vocab_digest.hexdigest(),
        # tokens whose bytes are not valid UTF-8 on their own: the replacement rules matter
        "lossy_tokens": [
            [i, t] for i, t in enumerate(ds.vocab_list) if i < 50257 and "�" in t
        ],
        "single_char_ids": ds._single_char_ids,
        "lists": lists,
        "masks": masks,
        "base_blocked_by_lang": base_by_lang,
        "bracket_allowed": bracket_allowed,
        "props": props,
    }

    texts = edge_cases() + random_texts(tok)
    rng = random.Random(1)
    sequences = [
        [rng.randrange(tok.encoding.n_vocab) for _ in range(rng.randint(1, 8))] for _ in range(300)
    ] + [[rng.randrange(50257) for _ in range(rng.randint(2, 6))] for _ in range(300)]
    traced = trace_texts(args.fixtures)
    tokenizer = {
        "encode": [[t, tok.encode(t)] for t in texts],
        "decode": [[s, tok.decode(s)] for s in sequences],
        # texts come from the traces at test time; only their ids' digests are committed
        "trace_texts": {
            "count": len(traced),
            "sha256": hashlib.sha256("\x00".join(traced).encode()).hexdigest(),
            # 8 hex digits per text are plenty to name the first text that differs
            "ids_sha256": [ids_digest(tok.encode(t))[:8] for t in traced],
        },
    }

    cases = run_cases(tok, ds)

    os.makedirs(args.out, exist_ok=True)
    for name, obj in [("decode_tables.json", tables), ("tokenizer.json", tokenizer), ("decode_cases.json", cases)]:
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, separators=(",", ":"))
        print(name, os.path.getsize(os.path.join(args.out, name)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
