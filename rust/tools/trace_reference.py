"""Record per-stage fixtures from the padding-fixed Python pipeline

The native runtime's parity tests replay these traces: every call into the
decode loop, the constraint state machine, alignment, DTW, the clean/* passes and
the speaker remap is logged with its inputs, its return value and, for functions
that mutate their arguments, the arguments after the call. Large arrays go to a
content-addressed blob store as `.npy` (only with `--save-blobs`; otherwise only
their sha256 is logged so a light trace still pins every tensor).

Run it through the GPU queue only. It mirrors `bench.py`: lang="en", BS=52, the
CUDA-graph padding fix, a 60 s untraced warmup on the list's first file and
`random.seed(0)` before each file, so its texts must equal the reference bench.

Layout:
  OUT/<stem>/events.jsonl   one JSON object per call, `seq` ordered
  OUT/<stem>/result.json    final text, segments and their sha256
  OUT/blobs/<sha20>.npy     arrays referenced by {"blob": sha20}
"""

import argparse
import functools
import hashlib
import inspect
import json
import math
import os
import random
import sys
import threading

import numpy as np
import soundfile as sf
import torch

# arrays at or under this many elements are inlined in the event
INLINE_MAX = 256
# per-function call cap per file, so hot helpers can't flood the trace
MAX_CALLS = 4000


class Tracer:
    """Shared event sink: one events file per traced audio file"""

    def __init__(self, out_dir, save_blobs):
        self.out_dir = out_dir
        self.blob_dir = os.path.join(out_dir, "blobs")
        os.makedirs(self.blob_dir, exist_ok=True)
        self.save_blobs = save_blobs
        self.lock = threading.Lock()
        self.local = threading.local()
        self.fh = None
        self.seq = 0
        self.counts = {}
        self.audio = None

    def start(self, stem, audio):
        d = os.path.join(self.out_dir, stem)
        os.makedirs(d, exist_ok=True)
        self.fh = open(os.path.join(d, "events.jsonl"), "w", encoding="utf-8")
        self.seq = 0
        self.counts = {}
        self.audio = audio

    def stop(self):
        if self.fh:
            self.fh.close()
        self.fh = None
        self.audio = None

    @property
    def active(self):
        return self.fh is not None

    def depth(self):
        return getattr(self.local, "depth", 0)

    def emit(self, ev):
        with self.lock:
            ev["seq"] = self.seq
            self.seq += 1
            self.fh.write(json.dumps(ev, ensure_ascii=False, allow_nan=False) + "\n")

    def take(self, name):
        n = self.counts.get(name, 0)
        self.counts[name] = n + 1
        return n < MAX_CALLS

    # encoding

    def array(self, a, save=True, extra=None):
        a = np.ascontiguousarray(a)
        out = {"__a": str(a.dtype), "shape": list(a.shape)}
        if extra:
            out.update(extra)
        if a.size <= INLINE_MAX:
            out["data"] = [_float(v) for v in a.reshape(-1).tolist()]
            return out
        sha = hashlib.sha256(a.tobytes()).hexdigest()[:20]
        out["blob"] = sha
        if save and self.save_blobs:
            path = os.path.join(self.blob_dir, sha + ".npy")
            if not os.path.exists(path):
                tmp = path.removesuffix(".npy") + f".{os.getpid()}.{threading.get_ident()}.tmp.npy"
                np.save(tmp, a, allow_pickle=False)
                os.replace(tmp, path)
            out["saved"] = True
        return out

    def tensor(self, t, save=True):
        if self.audio is not None and t.device.type == "cpu" and t.dtype == self.audio.dtype:
            try:
                same = t.untyped_storage().data_ptr() == self.audio.untyped_storage().data_ptr()
            except RuntimeError:
                same = False
            if same and t.dim() == 1 and t.stride(0) == 1:
                # a view of the file's own audio: the wav plus the range is enough
                return {"__audio": [t.storage_offset(), t.numel()]}
        x = t.detach()
        if x.dtype == torch.bfloat16:
            x = x.float()
        return self.array(x.cpu().numpy(), save, {"device": t.device.type})

    def enc(self, x, save=True):
        from whisperdrz.inference_utils import InferenceState

        if x is None or isinstance(x, (bool, int, str)):
            return x
        if isinstance(x, float):
            return _float(x)
        if isinstance(x, torch.Tensor):
            return self.tensor(x, save)
        if isinstance(x, np.ndarray):
            return self.array(x, save)
        if isinstance(x, np.generic):
            return _float(x.item())
        if isinstance(x, InferenceState):
            return {"__state": snap_state(x, self)}
        if isinstance(x, tuple):
            return {"__tuple": [self.enc(v, save) for v in x]}
        if isinstance(x, list):
            return [self.enc(v, save) for v in x]
        if isinstance(x, set):
            # keep CPython's iteration order, the code under test sees that order
            return {"__set": [self.enc(v, save) for v in x]}
        if isinstance(x, dict):
            if all(isinstance(k, str) for k in x):
                return {k: self.enc(v, save) for k, v in x.items()}
            return {"__items": [[self.enc(k, save), self.enc(v, save)] for k, v in x.items()]}
        if isinstance(x, (torch.dtype, torch.device)):
            return {"__repr": str(x)}
        return {"__type": type(x).__name__}

    def enc_after(self, x):
        """Re-encode only values a callee can mutate: states, arrays and containers"""
        from whisperdrz.inference_utils import InferenceState

        if isinstance(x, (InferenceState, np.ndarray, list, dict, set)):
            return self.enc(x)
        return None


def _float(v):
    if isinstance(v, float) and not math.isfinite(v):
        return {"__f": repr(v)}
    return v


# InferenceState fields that hold tensors or audio, logged elsewhere
STATE_SKIP = {"audio", "mel", "tokens", "initial_tokens", "speaker_hidden_states", "_last_hidden_state"}


def snap_state(s, tr):
    out = {}
    for k, v in vars(s).items():
        if k in STATE_SKIP:
            continue
        if isinstance(v, torch.Tensor):
            out[k] = {"__type": "Tensor"}
            continue
        out[k] = tr.enc(v, save=False)
    return out


def record(tr, owner, name, label=None, after=False, ret_save=True, ret=True, pre=None):
    """Replace `owner.name` with a recording wrapper; returns the original"""
    orig = getattr(owner, name)
    label = label or name
    try:
        sig = inspect.signature(orig)
    except (TypeError, ValueError):
        sig = None

    @functools.wraps(orig)
    def inner(*a, **kw):
        if not tr.active or not tr.take(label):
            return orig(*a, **kw)
        if sig is not None:
            try:
                bound = sig.bind(*a, **kw)
                named = dict(bound.arguments)
            except TypeError:
                named = {"args": list(a), **kw}
        else:
            named = {"args": list(a), **kw}
        ev = {"ev": label, "depth": tr.depth()}
        if pre is not None:
            named = pre(named)
        ev["args"] = {k: tr.enc(v) for k, v in named.items()}
        tr.local.depth = tr.depth() + 1
        try:
            out = orig(*a, **kw)
        except Exception as e:
            ev["raise"] = f"{type(e).__name__}: {e}"
            tr.emit(ev)
            raise
        finally:
            tr.local.depth = tr.depth() - 1
        if after:
            post = {}
            for k, v in named.items():
                e = tr.enc_after(v)
                if e is not None:
                    post[k] = e
            ev["after"] = post
        if ret:
            ev["ret"] = tr.enc(out, save=ret_save)
        tr.emit(ev)
        return out

    # module globals that imported the function by name resolve to the wrapper too
    for mod in list(sys.modules.values()):
        if mod is None or not getattr(mod, "__name__", "").startswith("whisperdrz"):
            continue
        if getattr(mod, name, None) is orig:
            setattr(mod, name, inner)
    setattr(owner, name, inner)
    return orig


def install_pad_fix():
    """Route CUDA-graph padding rows to a trash KV slot, as bench.py --pad-fix does"""
    import whisperdrz.model as M
    import whisperdrz.transcriber as T

    orig_create = M.WhisperDRZ.create_flash_kv

    def create(self, batch_size, max_tokens=None):
        return orig_create(self, batch_size + 1, max_tokens)

    M.WhisperDRZ.create_flash_kv = create
    orig_forward = T.BucketedCUDAGraphs.forward

    def forward(self, tokens, positions, slot_indices):
        B = tokens.shape[0]
        bucket = T._get_bucket(B, self.buckets)
        if bucket not in self.graphs:
            return orig_forward(self, tokens, positions, slot_indices)
        trash = self.self_kv[0].batch_size - 1
        self.static_tokens[bucket][:B].copy_(tokens)
        self.static_positions[bucket][:B].copy_(positions)
        self.static_slot_indices[bucket][:B].copy_(slot_indices)
        if B < bucket:
            self.static_tokens[bucket][B:].zero_()
            self.static_positions[bucket][B:].zero_()
            self.static_slot_indices[bucket][B:].fill_(trash)
            for kv in self.self_kv:
                kv.seq_lens[trash] = 0
        self.graphs[bucket].replay()
        return self.static_output[bucket][:B]

    T.BucketedCUDAGraphs.forward = forward


# whole modules whose module-level Python functions are recorded
PURE_MODULES = [
    "whisperdrz.inference_utils",
    "whisperdrz.speaker_clustering",
    "whisperdrz.tokenizer",
]
# hot per-character or per-token helpers: covered by their callers' records
SKIP_FUNCS = {
    "is_word_char",
    "text_has_word",
    "get_tokenizer",
    "init_for_model",
    "get_encoding",
}


def install(tr, transcriber):
    import whisperdrz
    import whisperdrz.decoding as D
    import whisperdrz.dtw_align as dtw
    import whisperdrz.transcriber as T

    # decode loop: the schedule (graph forward), the state machine, state updates
    record(tr, T.BucketedCUDAGraphs, "forward", label="graph_forward", ret_save=False)

    def decode_pre(named):
        # the repetition check reads all_tokens[slot, max(0, n-100):n] when n >= 25 and n % 10 == 0
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

    # slot lifecycle and alignment hand-offs
    for meth in (
        "_init_slot",
        "_prefill_slots",
        "_prepare_slot_for_alignment",
        "_batch_align_slots",
        "_dtw_fill_cur",
        "_finalize_slot_after_alignment",
    ):
        if hasattr(T.Transcriber, meth):
            record(tr, T.Transcriber, meth, after=True)

    # stage tensors: mel, encoder, prefill
    m = transcriber.model
    record(tr, transcriber.tomel, "forward", label="mel")
    record(tr, m.encoder, "forward", label="encoder")
    record(tr, m, "prefill_flash")
    record(tr, m.decoder, "forward", label="dtw_decoder_forward", ret=False)

    # DTW: the batched entry point, GPU prep, anchor bands (mutates its matrix) and the CPU DTW
    for name in ("fill_word_timings_batch", "_dtw_prep_batch", "_apply_anchor_bands", "_dtw_from_neg_matrix"):
        record(tr, dtw, name, after=name == "_apply_anchor_bands")

    # speaker remap: line audio ranges, ECAPA2 embeddings, clustering
    for name in ("extract_line_audio_segments", "_embed_audio_segment", "remap_speakers_per_line"):
        record(tr, T, name)

    # every module-level Python function in clean/* and the text helpers
    names = list(PURE_MODULES) + [
        n for n in sys.modules if n.startswith("whisperdrz.clean") and sys.modules[n] is not None
    ]
    for mod_name in sorted(set(names)):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for name, fn in list(vars(mod).items()):
            if not inspect.isfunction(fn) or fn.__module__ != mod_name or name in SKIP_FUNCS:
                continue
            record(tr, mod, name, label=f"{mod_name.removeprefix('whisperdrz.')}.{name}", after=True)

    # transcriber's own free text helpers
    for name in (
        "_strip_to_last_closing",
        "parse_to_segments",
    ):
        record(tr, T, name)

    # the remap crops long lines with the global random module
    for name in ("uniform", "randint"):
        orig = getattr(random, name)

        def wrap(*a, _orig=orig, _name=name):
            out = _orig(*a)
            if tr.active:
                tr.emit({"ev": f"random.{_name}", "depth": tr.depth(), "args": list(a), "ret": out})
            return out

        setattr(random, name, wrap)
    return whisperdrz


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="whisperdrz-large-v3.safetensors")
    p.add_argument("--src", required=True, help="frozen whisperdrz src dir to import")
    p.add_argument("--clip-s", type=float, default=None)
    p.add_argument("--save-blobs", action="store_true", help="write arrays over INLINE_MAX elements")
    args = p.parse_args()

    sys.path.insert(0, os.path.abspath(args.src))
    import whisperdrz

    assert os.path.abspath(whisperdrz.__file__).startswith(os.path.abspath(args.src)), whisperdrz.__file__
    # import every module the pipeline uses up front so record() can patch them all
    import whisperdrz.clean  # noqa: F401
    import whisperdrz.decoding  # noqa: F401
    import whisperdrz.dtw_align  # noqa: F401
    import whisperdrz.inference_utils  # noqa: F401
    import whisperdrz.speaker_clustering  # noqa: F401
    import whisperdrz.transcriber  # noqa: F401

    install_pad_fix()
    files = [line.strip() for line in open(args.list) if line.strip()]
    audios = []
    for f in files:
        a, sr = sf.read(f, dtype="float32", always_2d=True)
        assert sr == 16000, (f, sr)
        a = a.mean(1)
        if args.clip_s:
            a = a[: int(args.clip_s * 16000)]
        audios.append(torch.from_numpy(np.ascontiguousarray(a)))

    trn = whisperdrz.load_model(args.model, lang="en")
    trn.load_model()
    tr = Tracer(args.out, args.save_blobs)
    install(tr, trn)

    random.seed(0)
    trn.transcribe(audios[0][: 60 * 16000])
    torch.cuda.synchronize()

    summary = {"files": [], "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "src": args.src}
    for f, a in zip(files, audios, strict=True):
        stem = os.path.splitext(os.path.basename(f))[0] + (f"-clip{int(args.clip_s)}" if args.clip_s else "")
        random.seed(0)
        tr.start(stem, a)
        try:
            res = trn.transcribe(a)
            torch.cuda.synchronize()
        finally:
            n_events = tr.seq
            tr.stop()
        blob = json.dumps({"text": res.text, "segments": res.segments}, sort_keys=True, ensure_ascii=False)
        out = {
            "file": f,
            "clip_s": args.clip_s,
            "samples": len(a),
            "text": res.text,
            "segments": res.segments,
            "sha256": hashlib.sha256(blob.encode()).hexdigest(),
            "events": n_events,
            "raw_chunks": tr.enc([{k: v for k, v in c.items() if k != "audio"} for c in res.raw_chunks], save=False),
        }
        with open(os.path.join(args.out, stem, "result.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        summary["files"].append({"stem": stem, "sha256": out["sha256"], "events": n_events})
        print(json.dumps(summary["files"][-1]), flush=True)

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
