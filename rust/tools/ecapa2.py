"""ECAPA2 (Jenthe/ECAPA2, TorchScript `ecapa2.pt`) for the native port: weight export,
a functional re-implementation, and the reference data the Rust GPU tests read

`rust/docs/ecapa2.md` describes the architecture this file re-implements from the
TorchScript code. Subcommands:

  export OUT_DIR   every parameter and buffer as FP32 to OUT_DIR/ecapa2.safetensors
                   under its TorchScript name (CPU only)
  check            the functional forward against the TorchScript one on the CPU, bit
                   for bit, on recorded segments: proof that the architecture is right
  ref OUT_DIR      GPU only, run through the queue: the CUDA kernels torch dispatches,
                   per-layer FP32 truth (TF32 off) for three segments, per-layer error
                   of torch as the reference runs it (cuDNN TF32 on, matmul TF32 off),
                   and every recorded segment's audio and embeddings

Run from `rust/` with the venv:
  CUDA_VISIBLE_DEVICES="" ~/code/whisperdrz/.venv/bin/python -I tools/ecapa2.py export ~/.cache/whisperdrz/ecapa2
  CUDA_VISIBLE_DEVICES="" ~/code/whisperdrz/.venv/bin/python -I tools/ecapa2.py check
  ~/code/whisperdrz/.venv/bin/python -I tools/ecapa2.py ref ~/.cache/whisperdrz/ecapa2/ref
"""

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

MODEL = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Jenthe--ECAPA2/snapshots/*/ecapa2.pt"))
FIXTURES = os.environ.get("WHISPERDRZ_FIXTURES", os.path.expanduser("~/.cache/whisperdrz/fixtures/ref-a783291"))
SETS = ["short3", "e21c600"]
# per-layer dumps: the shortest, a median and the longest recorded segment
LAYER_SEGMENTS = [("short3", "willh", 2935), ("short3", "ikgcq", 3641), ("e21c600", "4360674-clip600", 11570)]

# the 2D trunk in order: (block, stride of its first conv along frequency, residual)
# residual is "id" (x + block(x)), a downsample module name, or None (channels change)
TRUNK = [
    ("conv_1", 1, None),
    ("conv_2", 1, "id"),
    ("conv_17", 1, "id"),
    ("conv_3", 2, "downsample_1"),
    ("conv_4", 1, "id"),
    ("conv_5", 1, "id"),
    ("conv_18", 1, "id"),
    ("conv_6", 2, "downsample_2"),
    ("conv_7", 1, "id"),
    ("conv_8", 1, None),
    ("conv_19", 1, "id"),
    ("conv_9", 2, "downsample_3"),
    ("conv_10", 1, "id"),
    ("conv_11", 1, "id"),
    ("conv_20", 1, "id"),
    ("conv_12", 2, "downsample_4"),
    ("conv_13", 1, "id"),
    ("conv_14", 1, "id"),
    ("conv_15", 1, "id"),
    ("conv_16", 1, "id"),
]
TAPS = ["spec"] + [b for b, _, _ in TRUNK] + ["tdnn_1", "tdnn_2", "dense_1", "pool", "embedding"]


def load_script(device):
    assert len(MODEL) == 1, MODEL
    m = torch.jit.load(MODEL[0], map_location=device)
    m.eval()
    return m


def weights(m):
    """Parameters and buffers by TorchScript name, without the BN step counters"""
    out = {n: p.detach() for n, p in m.named_parameters()}
    out.update({n: b.detach() for n, b in m.named_buffers() if not n.endswith("num_batches_tracked")})
    return out


def forward(W, x, taps=None):
    """ECAPA2's `forward(x, labels="embedding")` with the TorchScript's ops in its order

    x: raw 16 kHz audio (1, T) FP32. Returns the unnormalized (1, 192) embedding; `taps`
    (a dict) receives each block's output
    """
    tap = (lambda k, v: taps.__setitem__(k, v.detach().clone())) if taps is not None else (lambda k, v: None)

    def bn(t, p):
        return F.batch_norm(t, W[p + ".running_mean"], W[p + ".running_var"], W[p + ".weight"], W[p + ".bias"], False, 0.1, 1e-5)

    def fc(y, p):
        y = F.linear(y, W[p + ".0.weight"], W[p + ".0.bias"])
        y = bn(F.relu(y), p + ".2")
        return torch.sigmoid(F.linear(y, W[p + ".3.weight"], W[p + ".3.bias"]))

    def conv_block(t, p, stride):
        for i, s in ((0, [stride, 1]), (3, [1, 1]), (6, [1, 1])):
            t = F.conv2d(t, W[f"{p}.{i}.weight"], W[f"{p}.{i}.bias"], s, [1, 1], [1, 1])
            t = bn(F.relu(t), f"{p}.{i + 2}")
        b, c = t.shape[:2]
        y = F.adaptive_avg_pool2d(t, [1, 1]).view(b, c)
        return t * fc(y, p + ".9.fc").view(b, c, 1, 1)

    def downsample(t, p):
        return bn(F.conv2d(t, W[p + ".0.weight"], None, [2, 1], [0, 0], [1, 1]), p + ".1")

    def conv1(t, p, dilation=1):
        return F.conv1d(t, W[p + ".weight"], W[p + ".bias"], [1], [0], [dilation])

    # front end: torchaudio Spectrogram(n_fft=511, win_length=400, hop=160, power=2,
    # centered, reflect), log(x + 1e-6), minus the mean over time
    spec = torchaudio.functional.spectrogram(
        x, 0, W["pre_processing.0.window"], 511, 160, 400, 2.0, False, True, "reflect", True
    )
    spec = torch.log(spec + 1e-6)
    spec = spec - torch.mean(spec, [-1], True)
    tap("spec", spec[0])
    h = torch.squeeze(spec, 1).to(torch.float32).unsqueeze(1)

    for block, stride, res in TRUNK:
        y = conv_block(h, block, stride)
        if res == "id":
            y = y + h
        elif res is not None:
            y = y + downsample(h, res)
        h = y
        tap(block, h[0])

    b, c, f, t = h.shape
    h = conv1(h.reshape(b, c * f, t), "tdnn_1.0")
    h = bn(F.relu(h), "tdnn_1.2")
    tap("tdnn_1", h[0])

    # tdnn_2: 1x1 conv, Res2Net (8 groups of 128, k3 dilation 2, reflect padding 2),
    # 1x1 conv, SE over time, plus the residual
    r = bn(F.relu(conv1(h, "tdnn_2.0")), "tdnn_2.2")
    chunks = torch.chunk(r, 8, 1)
    outs = [chunks[0]]
    for i in range(1, 8):
        inp = chunks[i] if i == 1 else chunks[i] + outs[i - 1]
        y = conv1(F.pad(inp, [2, 2], "reflect"), f"tdnn_2.3.conv_filters.{i - 1}", 2)
        outs.append(bn(F.relu(y), f"tdnn_2.3.batch_norms.{i - 1}"))
    r = torch.cat(outs, 1)
    r = bn(F.relu(conv1(r, "tdnn_2.4")), "tdnn_2.6")
    mean = torch.mean(r.to(torch.float32), [2]).to(r.dtype)
    r = r * fc(mean, "tdnn_2.7.fc")[:, :].unsqueeze(2).expand_as(r)
    h = r + h
    tap("tdnn_2", h[0])

    h = F.relu(conv1(h, "dense_1.0"))
    tap("dense_1", h[0])

    # attentive statistics pooling with global context
    m = torch.mean(h, [2], True).expand_as(h)
    s = torch.sqrt(torch.var(h, [2], True, True) + 1e-5).expand_as(h)
    a = conv1(torch.cat([h, m, s], 1), "pooling_1.0.attention.0")
    a = bn(F.relu(a), "pooling_1.0.attention.2")
    a = conv1(a, "pooling_1.0.attention.3")
    w = F.softmax(a, 2)
    xs = w * h
    mu = torch.sum(xs, [2])
    sigma = torch.sqrt(torch.abs(torch.sum(xs * h, [2]) - torch.pow(mu, 2)) + 1e-5)
    pool = bn(torch.cat([mu, sigma], 1), "pooling_1.1")
    tap("pool", pool[0])

    emb = F.linear(pool, W["dense_2.0.weight"], W["dense_2.0.bias"])
    tap("embedding", emb[0])
    return emb


def file_audio(set_name, stem):
    res = json.load(open(os.path.join(FIXTURES, set_name, stem, "result.json")))
    a, sr = sf.read(res["file"], dtype="float32", always_2d=True)
    assert sr == 16000
    a = a.mean(1)
    if res.get("clip_s"):
        a = a[: int(res["clip_s"] * 16000)]
    assert len(a) == res["samples"], (len(a), res["samples"])
    return np.ascontiguousarray(a)


def segments():
    """Every recorded `_embed_audio_segment` call: (set, stem, seq, audio, recorded)"""
    out = []
    for set_name in SETS:
        summary = json.load(open(os.path.join(FIXTURES, set_name, "summary.json")))
        for f in summary["files"]:
            stem = f["stem"]
            audio = file_audio(set_name, stem)
            for line in open(os.path.join(FIXTURES, set_name, stem, "events.jsonl")):
                ev = json.loads(line)
                if ev["ev"] != "_embed_audio_segment":
                    continue
                start, n = ev["args"]["audio_tensor"]["__audio"]
                rec = np.array(ev["ret"]["data"], dtype=np.float32)
                assert rec.shape == (192,)
                out.append((set_name, stem, ev["seq"], audio[start : start + n], rec))
    assert len(out) == 76, len(out)
    return out


def normalize(e):
    return F.normalize(e.squeeze(0).cpu().float(), dim=-1)


def cmd_export(out_dir):
    from safetensors.torch import save_file

    m = load_script("cpu")
    W = {k: v.contiguous() for k, v in weights(m).items()}
    assert all(v.dtype == torch.float32 for v in W.values())
    sha = hashlib.sha256(open(MODEL[0], "rb").read()).hexdigest()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ecapa2.safetensors")
    save_file(W, path, metadata={"source": "Jenthe/ECAPA2 ecapa2.pt", "sha256": sha})
    print(f"wrote {len(W)} tensors to {path}")


def cmd_check():
    m = load_script("cpu")
    W = weights(m)
    segs = segments()
    picks = [s for s in segs if (s[0], s[1], s[2]) in LAYER_SEGMENTS]
    assert len(picks) == 3
    with torch.inference_mode():
        for set_name, stem, seq, audio, _ in picks:
            x = torch.from_numpy(audio).unsqueeze(0)
            ours, ref = forward(W, x), m(x)
            same = torch.equal(ours, ref)
            print(f"{set_name}/{stem} #{seq}: T={x.shape[1]} functional == TorchScript: {same}, max-abs {(ours - ref).abs().max().item():.3e}")
            assert same


def stats(a, truth):
    d = (a.double() - truth.double()).abs()
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item(), "truth_max_abs": truth.abs().max().item()}


def set_mode(tf32):
    # the reference never sets these, so it runs torch's defaults: cuDNN TF32 on, matmul off
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.allow_tf32 = False


def cmd_ref(out_dir):
    from torch.profiler import ProfilerActivity, profile

    dev = torch.device("cuda")
    os.makedirs(out_dir, exist_ok=True)
    m = load_script(dev)
    W = weights(m)
    segs = segments()

    # 1. what torch dispatches for one call, as the reference ran (TorchScript, defaults),
    # after warm-up runs so the profiling executor has specialized the graph
    set_mode(True)
    x = torch.from_numpy(segs[0][3]).unsqueeze(0).to(dev)
    with torch.inference_mode():
        for _ in range(3):
            m(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            m(x)
            torch.cuda.synchronize()
    kernels = {}
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            kernels[e.name] = kernels.get(e.name, 0) + 1
    with open(os.path.join(out_dir, "dispatch.txt"), "w") as fh:
        fh.write(f"torch {torch.__version__} cudnn {torch.backends.cudnn.version()} {torch.cuda.get_device_name()}\n")
        fh.write(f"TorchScript forward, T={x.shape[1]}, cudnn.allow_tf32=True, matmul.allow_tf32=False\n")
        for name, n in sorted(kernels.items(), key=lambda kv: -kv[1]):
            fh.write(f"{n:4d}  {name}\n")

    # 2. every segment: audio, recorded embedding, TorchScript (defaults), functional in
    # reference mode and the FP32 truth, all normalized as `_embed_audio_segment` does
    seg_dir = os.path.join(out_dir, "segments")
    os.makedirs(seg_dir, exist_ok=True)
    manifest = {"segments": [], "layers": {}, "taps": TAPS, "torch": torch.__version__}
    with torch.inference_mode():
        for set_name, stem, seq, audio, rec in segs:
            name = f"{set_name}-{stem}-{seq}"
            np.save(os.path.join(seg_dir, name + ".npy"), audio)
            x = torch.from_numpy(audio).unsqueeze(0).to(dev)
            set_mode(True)
            script = normalize(m(x))
            func_ref = normalize(forward(W, x))
            set_mode(False)
            truth = normalize(forward(W, x))
            r = torch.from_numpy(rec)
            manifest["segments"].append(
                {
                    "name": name,
                    "set": set_name,
                    "stem": stem,
                    "seq": seq,
                    "samples": len(audio),
                    "audio": f"segments/{name}.npy",
                    "recorded": rec.tolist(),
                    "truth": truth.tolist(),
                    "cos_script_recorded": F.cosine_similarity(script, r, 0).item(),
                    "cos_reference_mode_recorded": F.cosine_similarity(func_ref, r, 0).item(),
                    "cos_truth_recorded": F.cosine_similarity(truth, r, 0).item(),
                    "script_equals_recorded": bool(torch.equal(script, r)),
                }
            )

        # 3. per-layer truth for three segments, and torch's reference-mode error per layer
        for set_name, stem, seq, audio, _ in segs:
            if (set_name, stem, seq) not in LAYER_SEGMENTS:
                continue
            name = f"{set_name}-{stem}-{seq}"
            x = torch.from_numpy(audio).unsqueeze(0).to(dev)
            set_mode(False)
            truth = {}
            forward(W, x, truth)
            set_mode(True)
            ref = {}
            forward(W, x, ref)
            layer_dir = os.path.join(out_dir, "layers", name)
            os.makedirs(layer_dir, exist_ok=True)
            rows = {}
            for k in TAPS:
                np.save(os.path.join(layer_dir, k + ".npy"), np.ascontiguousarray(truth[k].cpu().numpy()))
                rows[k] = {"shape": list(truth[k].shape), "reference_mode": stats(ref[k], truth[k])}
            manifest["layers"][name] = rows

    set_mode(True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    worst = min(s["cos_truth_recorded"] for s in manifest["segments"])
    print(f"wrote {len(manifest['segments'])} segments; min cos(truth, recorded) {worst:.6f}")
    print(f"TorchScript equals recorded on {sum(s['script_equals_recorded'] for s in manifest['segments'])} segments")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export").add_argument("out")
    sub.add_parser("check")
    sub.add_parser("ref").add_argument("out")
    args = p.parse_args()
    if args.cmd == "export":
        cmd_export(os.path.expanduser(args.out))
    elif args.cmd == "check":
        cmd_check()
    else:
        cmd_ref(os.path.expanduser(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
