"""Per-layer and end-to-end torch references for the native encoder

usage: encoder_layers.py --src SNAPSHOT_SRC --model CKPT OUT_DIR

Reads recorded `encoder` inputs from the reference traces and writes, for the native
encoder test (`tests/gpu_encoder.rs`):

  layers/mel.npy          FP16 input: batch items 0 and 4 of short3/nctdh's first call
  layers/f16_<name>.npy   torch FP16 (the reference: FP16 weights, SDPA flash backend)
  layers/f32_<name>.npy   torch FP32 on the same FP16 weights and input: the truth chain
  layers/f16eff_<name>.npy  torch FP16 with SDPA's memory-efficient backend instead of
                          flash, a change exact in real arithmetic: a second FP16 sample
                          that shows how far two equally valid FP16 runs drift apart
  layers/f16gemm_<name>.npy torch FP16 with both convs lowered to unfold + F.linear (an
                          im2col GEMM, as the native stem runs them) instead of cuDNN:
                          a third sample, exact in real arithmetic
  layers/iso_<name>.npy   FP32 layer <name> on torch-FP16's previous layer output, the
                          truth for a layer checked in isolation
  e2e/<set>-<stem>-<seq>.f32.npy   FP32 output on a whole recorded batch
  manifest.json

<name> is `stem` (after the positional embedding), `block00`..`block31` and `ln_post`.
Layer outputs are captured with forward hooks on the reference AudioEncoder: a pre-hook
on block 0 for the stem, a hook on every block and on ln_post. FP32 runs with TF32 off
and SDPA's math backend, so nothing in the truth rounds below FP32. Needs the GPU, so
run it through the queue (`scripts/gpu-test.sh` does); skips when the manifest of this
version is already there
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

VERSION = 3
LAYER_CASE = ("short3", "nctdh", 0)  # trace set, file stem, encoder call index
LAYER_ITEMS = [0, 4]
E2E_CASES = [("short3", "nctdh", 0), ("e21c600", "4360674-clip600", 0)]


def fixtures_root():
    return os.environ.get(
        "WHISPERDRZ_FIXTURES",
        os.path.expanduser("~/.cache/whisperdrz/fixtures/ref-a783291"),
    )


def encoder_call(trace_set, stem, index):
    """The recorded input and output arrays of one `encoder` call"""
    root = os.path.join(fixtures_root(), trace_set)
    calls = []
    with open(os.path.join(root, stem, "events.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            ev = json.loads(line)
            if ev.get("ev") == "encoder":
                calls.append(ev)
    calls.sort(key=lambda e: e["seq"])
    ev = calls[index]

    def load(v):
        assert v.get("saved"), f"{trace_set}/{stem} encoder #{ev['seq']}: blob not saved"
        return np.load(os.path.join(root, "blobs", v["blob"] + ".npy"))

    return ev["seq"], load(ev["args"]["x"]), load(ev["ret"])


def capture(encoder):
    """Registers hooks that store each layer's output; returns (outputs, handles)"""
    out = {}
    handles = [
        encoder.blocks[0].register_forward_pre_hook(
            lambda m, args: out.__setitem__("stem", args[0].detach().clone())
        )
    ]
    for i, block in enumerate(encoder.blocks):
        handles.append(
            block.register_forward_hook(
                lambda m, a, y, i=i: out.__setitem__(f"block{i:02d}", y.detach().clone())
            )
        )
    handles.append(
        encoder.ln_post.register_forward_hook(
            lambda m, a, y: out.__setitem__("ln_post", y.detach().clone())
        )
    )
    return out, handles


class GemmConv(torch.nn.Module):
    """A Conv1d run as unfold + F.linear on the conv's own weight and bias"""

    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, x):
        c = self.conv
        k, s, p = c.kernel_size[0], c.stride[0], c.padding[0]
        cols = torch.nn.functional.pad(x, (p, p)).unfold(2, k, s)  # [B, C, T, k]
        cols = cols.permute(0, 2, 1, 3).reshape(x.shape[0], cols.shape[2], -1)
        w = c.weight.reshape(c.weight.shape[0], -1)
        return torch.nn.functional.linear(cols, w, c.bias).transpose(1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("out")
    args = ap.parse_args()

    manifest_path = os.path.join(args.out, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            if json.load(fh).get("version") == VERSION:
                print(f"encoder references already in {args.out}")
                return

    sys.path.insert(0, os.path.abspath(args.src))
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from whisperdrz.model import WhisperDRZ
    from whisperdrz.utils import flash_kernel

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.makedirs(os.path.join(args.out, "layers"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "e2e"), exist_ok=True)

    def save(rel, t):
        np.save(os.path.join(args.out, rel), t.detach().cpu().numpy())
        return rel

    # the reference model, as the transcriber loads and runs it
    enc16 = WhisperDRZ.from_pretrained_inf(args.model).cuda().encoder
    # the same FP16 weights widened to FP32
    enc32 = WhisperDRZ.from_pretrained(args.model).cuda().eval().encoder
    torch.cuda.empty_cache()

    def run16(x):
        with torch.inference_mode(), flash_kernel():
            return enc16(x)

    def run32(x):
        with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
            return enc32(x.float())

    manifest = {"version": VERSION, "e2e": []}
    for trace_set, stem, index in E2E_CASES:
        seq, mel, recorded = encoder_call(trace_set, stem, index)
        x = torch.from_numpy(mel).cuda()
        y16 = run16(x)
        diff = (y16.float() - torch.from_numpy(recorded).cuda().float()).abs().max().item()
        # FP32 in chunks of 4, so the math backend's score matrices stay small
        y32 = torch.cat([run32(x[i : i + 4]) for i in range(0, x.shape[0], 4)])
        name = f"e2e/{trace_set}-{stem}-{seq}.f32.npy"
        save(name, y32)
        manifest["e2e"].append(
            {
                "trace": trace_set,
                "stem": stem,
                "seq": seq,
                "f32": name,
                "f16_rerun_max_diff": diff,
            }
        )
        print(f"{trace_set}/{stem} #{seq}: B={x.shape[0]}, torch FP16 rerun vs recorded max diff {diff}")

    trace_set, stem, index = LAYER_CASE
    seq, mel, _ = encoder_call(trace_set, stem, index)
    x = torch.from_numpy(mel[LAYER_ITEMS]).cuda().contiguous()
    save("layers/mel.npy", x)

    out16, handles = capture(enc16)
    run16(x)
    for h in handles:
        h.remove()
    out16eff, handles = capture(enc16)
    with torch.inference_mode(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        enc16(x)
    for h in handles:
        h.remove()
    convs = (enc16.conv1, enc16.conv2)
    enc16.conv1, enc16.conv2 = GemmConv(convs[0]), GemmConv(convs[1])
    out16gemm, handles = capture(enc16)
    run16(x)
    for h in handles:
        h.remove()
    enc16.conv1, enc16.conv2 = convs
    out32, handles = capture(enc32)
    run32(x)
    for h in handles:
        h.remove()

    names = ["stem"] + [f"block{i:02d}" for i in range(len(enc16.blocks))] + ["ln_post"]
    with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
        prev = None
        for name in names:
            if name == "stem":
                iso = out32["stem"]
            elif name == "ln_post":
                iso = enc32.ln_post(prev.float())
            else:
                iso = enc32.blocks[int(name[5:])](prev.float())
            save(f"layers/f16_{name}.npy", out16[name])
            save(f"layers/f16eff_{name}.npy", out16eff[name])
            save(f"layers/f16gemm_{name}.npy", out16gemm[name])
            save(f"layers/f32_{name}.npy", out32[name])
            save(f"layers/iso_{name}.npy", iso)
            prev = out16[name]

    manifest["layers"] = {
        "trace": trace_set,
        "stem": stem,
        "seq": seq,
        "items": LAYER_ITEMS,
        "mel": "layers/mel.npy",
        "names": names,
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"wrote encoder references for {len(names)} layers to {args.out}")


if __name__ == "__main__":
    main()
