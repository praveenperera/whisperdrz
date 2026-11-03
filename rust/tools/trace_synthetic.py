"""Trace the real Python scheduler on the CPU with a fake model, for slot refills

The reference traces never refill a freed slot (every file fits in BS=52
slots at once), so the CPython `set` order of `empty_slots`, refills and more
window continuations go unexercised there. This runs `Transcriber.transcribe`
from the frozen snapshot with a small BS and a stand-in model on the CPU, and
records it under the same labels as `trace_reference.py`, so the Rust replay
test can check it the same way:

- encoder: seeded random fp16 output, `n_audio_state` 8 to keep blobs small
- prefill and decode steps: seeded random fp16 logits (normal, with a random
  boost to `<|endoftext|>` so windows end early and continue), KV lengths
  tracked like `FlashKVCache.seq_lens`
- DTW: the decoder forward is a no-op and cross-attention is seeded random
  noise; the real `_dtw_prep_batch` and CPU DTW run on it
- speaker remap: seeded random unit embeddings per line crop
- mel: the real `LogMelSpectrogram` on synthetic noise

Decoding, alignment, post-processing and the remap all run the real Python
code. No GPU is used.

Usage: CUDA_VISIBLE_DEVICES= python -I rust/tools/trace_synthetic.py --src SNAPSHOT_SRC --out DIR

The replay test reads two sets from ~/.cache/whisperdrz/fixtures/synth-a783291
(a sibling of the reference trace root, which must hold reference traces only):
  --out .../synth-a783291/synth-bs4                                  (default flags)
  --out .../synth-a783291/synth-bs6-lockstep --bs 6 --seconds 600.5 --lockstep
"""

import argparse
import hashlib
import json
import os
import random
import sys

import numpy as np
import torch

# fp16 logits scale and how often (and how hard) a row boosts end-of-text
LOGIT_SCALE = 2.0
EOT_PROB = 0.03
EOT_BOOST = 12.0
N_AUDIO_STATE = 8
EOT = 50257
VOCAB = 51866


class Rng:
    """One seeded stream per stand-in, so changing one doesn't move the others.
    With `lockstep` every row of a call gets the same logits, so slots that start
    together decode alike and free up together"""

    lockstep = False

    def __init__(self, seed):
        self.g = np.random.default_rng(seed)

    def logits(self, rows, positions=1):
        n = 1 if Rng.lockstep else rows
        x = self.g.standard_normal((n, positions, VOCAB)).astype(np.float32) * LOGIT_SCALE
        boost = self.g.random(n) < EOT_PROB
        x[boost, -1, EOT] += EOT_BOOST
        if Rng.lockstep:
            x = np.repeat(x, rows, axis=0)
        return torch.from_numpy(x.astype(np.float16))


class Dims:
    n_vocab = VOCAB
    n_mels = 128
    n_audio_state = N_AUDIO_STATE
    n_text_state = 1280
    n_text_layer = 32
    n_text_head = 20


class FakeKV:
    """`FlashKVCache` reduced to the per-slot lengths the scheduler reads"""

    def __init__(self, n):
        self.batch_size = n
        self.seq_lens = torch.zeros(n, dtype=torch.int32)

    def reset(self, slots):
        if slots is None:
            self.seq_lens.zero_()
        else:
            self.seq_lens[slots] = 0


class Encoder:
    def __init__(self, rng):
        self.rng = rng

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        n = x.shape[0]
        return torch.from_numpy(self.rng.g.standard_normal((n, 1500, N_AUDIO_STATE)).astype(np.float16))


class Decoder:
    """The DTW forward: remembers the batch shape for the attention stand-in"""

    last_shape = None

    def __call__(self, x, xa):
        return self.forward(x, xa)

    def forward(self, x, xa):
        Decoder.last_shape = tuple(x.shape)


class FakeModel:
    def __init__(self, tokenizer, rng):
        self.dims = Dims()
        self.is_multilingual = True
        self.tokenizer = tokenizer
        self.encoder = Encoder(rng["encoder"])
        self.decoder = Decoder()
        self.device = torch.device("cpu")
        self.dtype = torch.float16
        self.rng = rng["prefill"]

    def reset_flash_kv(self, self_kv, cross_kv, slot_indices=None):
        for s_kv, c_kv in zip(self_kv, cross_kv, strict=True):
            s_kv.reset(slot_indices)
            c_kv.reset(slot_indices)

    def prefill_flash(self, tokens, xa, self_kv, cross_kv, slot_indices, return_hidden_states=False):
        n, t = tokens.shape
        self_kv[0].seq_lens[slot_indices] = t
        return self.rng.logits(n, t)


class FakeGraphs:
    """`BucketedCUDAGraphs.forward`: one step, every row appends a KV position"""

    def __init__(self, self_kv, rng):
        self.self_kv = self_kv
        self.rng = rng

    def forward(self, tokens, positions, slot_indices):
        self.self_kv[0].seq_lens[slot_indices] += 1
        return self.rng.logits(tokens.shape[0])


class FakeCapture:
    def __init__(self, model, layers=None):
        self.rng = FakeCapture.rng

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def compute_attention(self, head_mask):
        b, t = Decoder.last_shape
        h = int(head_mask.sum())
        w = self.rng.g.standard_normal((b, h, t, 1500)).astype(np.float32)
        return torch.from_numpy(w)


def build(args, rng):
    import whisperdrz.dtw_align as D
    import whisperdrz.transcriber as T
    from whisperdrz.audio import LogMelSpectrogram
    from whisperdrz.decoding import DecoderState
    from whisperdrz.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(True, num_languages=100, language="en")
    model = FakeModel(tokenizer, rng)
    trn = T.Transcriber(None, BS=args.bs, lang="en", model=model)

    # load_model() without a device: the same attributes, CPU tensors
    n = trn.n_slots
    trn.use_cuda = False
    trn.device = torch.device("cpu")
    trn.dtype = torch.float16
    trn.tomel = LogMelSpectrogram(n_mels=128)
    trn.max_seq_len = trn.max_tokens
    trn.self_kv, trn.cross_kv = [FakeKV(n + 1)], [FakeKV(n + 1)]
    trn.ds = DecoderState(tokenizer, device="cpu")
    trn.ds.init_gpu_tensors()
    trn.all_tokens = torch.zeros(n, trn.max_seq_len, dtype=torch.long)
    trn.seq_lens = torch.zeros(n, dtype=torch.long)
    trn.batch_idx = torch.arange(n)
    trn.xa_cache = torch.zeros(n, 1500, N_AUDIO_STATE, dtype=torch.float16)
    trn.slot_tensor_cache = [torch.tensor([i], dtype=torch.long) for i in range(n)]
    trn._slot_indices_buffer = torch.arange(n, dtype=torch.long)
    trn.prefill_stream = None
    trn.seconds_processed = 0.0
    trn._max_tokens_reached = 0
    trn.cuda_graphs = FakeGraphs(trn.self_kv, rng["decode"])
    # two alignment heads in the last layer
    mask = torch.zeros(32, 20, dtype=torch.bool)
    mask[31, 0] = mask[31, 1] = True
    trn._dtw_head_mask = mask
    trn.speaker_remap_model = torch.nn.Linear(1, 1)
    trn.speaker_remap_model_name = "ecapa2"
    trn.loaded = True

    FakeCapture.rng = rng["attention"]
    D.CrossAttnCapture = FakeCapture
    # `_dtw_prep_batch` copies into pinned memory and syncs the device; on the CPU
    # a plain buffer and no sync do the same
    D._pinned_host_buffer = lambda shape, dtype: torch.empty(shape, dtype=dtype)
    torch.cuda.synchronize = lambda *a, **k: None

    emb_rng = rng["embed"]

    def embed(audio_tensor, model, device):
        v = torch.from_numpy(emb_rng.g.standard_normal(192).astype(np.float32))
        return torch.nn.functional.normalize(v, dim=-1)

    T._embed_audio_segment = embed
    return trn


def install(tr, trn):
    """The scheduler's labels from `trace_reference.install`, minus the text helpers"""
    import whisperdrz.decoding as D
    import whisperdrz.dtw_align as dtw
    import whisperdrz.transcriber as T
    from trace_reference import record

    record(tr, FakeGraphs, "forward", label="graph_forward", ret_save=False)

    def decode_pre(named):
        at = named.get("all_tokens_gpu")
        if at is not None and named.get("slot_indices") is not None:
            rows = {}
            for slot, n in zip(named["slot_indices"], named["token_counts"], strict=True):
                if n >= 25 and n % 10 == 0:
                    rows[str(slot)] = at[slot, max(0, n - 100) : n].tolist()
            named["all_tokens_gpu"] = {"__rows": rows}
        return named

    record(tr, D, "decode_batch", after=True, pre=decode_pre)
    record(tr, D, "update_state_after_decode", after=True)
    for meth in (
        "_init_slot",
        "_prefill_slots",
        "_prepare_slot_for_alignment",
        "_batch_align_slots",
        "_finalize_slot_after_alignment",
    ):
        record(tr, T.Transcriber, meth, after=True)

    record(tr, trn.tomel, "forward", label="mel")
    record(tr, trn.model.encoder, "forward", label="encoder")
    record(tr, trn.model, "prefill_flash")
    record(tr, trn.model.decoder, "forward", label="dtw_decoder_forward", ret=False)
    record(tr, dtw, "fill_word_timings_batch")

    def drop_weights(named):
        # the random attention is regenerated from its seed; keep the blob store small
        named.pop("full_weights", None)
        return named

    record(tr, dtw, "_dtw_prep_batch", pre=drop_weights)
    record(tr, T, "_embed_audio_segment")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="frozen whisperdrz src dir to import")
    p.add_argument("--out", required=True)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--seconds", type=float, default=442.3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--lockstep", action="store_true", help="same logits for every row of a call")
    args = p.parse_args()
    Rng.lockstep = args.lockstep
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "run with CUDA_VISIBLE_DEVICES="

    sys.path.insert(0, os.path.abspath(args.src))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import whisperdrz

    assert os.path.abspath(whisperdrz.__file__).startswith(os.path.abspath(args.src)), whisperdrz.__file__
    import whisperdrz.clean  # noqa: F401
    import whisperdrz.transcriber  # noqa: F401
    from trace_reference import Tracer

    rng = {k: Rng([args.seed, i]) for i, k in enumerate(["encoder", "prefill", "decode", "attention", "embed", "audio"])}
    trn = build(args, rng)
    tr = Tracer(args.out, save_blobs=True)
    install(tr, trn)

    n = int(args.seconds * 16000)
    audio = torch.from_numpy((rng["audio"].g.standard_normal(n) * 0.1).astype(np.float32))
    stem = f"noise-bs{args.bs}" + ("-lockstep" if args.lockstep else "")
    random.seed(0)
    tr.start(stem, audio)
    try:
        res = trn.transcribe(audio)
    finally:
        n_events = tr.seq
        tr.stop()

    with open(os.path.join(args.out, stem, "events.jsonl"), encoding="utf-8") as fh:
        raised = [line for line in fh if '"raise"' in line]
    assert not raised, f"{len(raised)} recorded calls raised, first: {raised[0][:300]}"

    blob = json.dumps({"text": res.text, "segments": res.segments}, sort_keys=True, ensure_ascii=False)
    out = {
        "file": None,
        "synthetic": vars(args),
        "samples": n,
        "text": res.text,
        "segments": res.segments,
        "sha256": hashlib.sha256(blob.encode()).hexdigest(),
        "events": n_events,
        "raw_chunks": tr.enc([{k: v for k, v in c.items() if k != "audio"} for c in res.raw_chunks], save=False),
    }
    with open(os.path.join(args.out, stem, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    summary = {"files": [{"stem": stem, "sha256": out["sha256"], "events": n_events}], "bs": args.bs, "src": args.src}
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary["files"][0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
