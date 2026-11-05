"""Torch references for the native decoder attention kernels

usage: attn_ref.py OUT_DIR --src SNAPSHOT_SRC --weights CKPT [--trace TRACE_SET_DIR]

Writes OUT_DIR/cases.json and one .npy per array:

- synthetic cases: random FP16 caches with ragged lengths, out-of-order slots and
  CUDA-graph padding rows on the trash slot, run through `flash_attn_with_kvcache` (the
  reference op, FP16) and through an FP32 torch attention on the same FP16 inputs (the
  truth both are scored against). Cache positions at or past a slot's length are zero
  here; the Rust test overwrites them with NaN before running its kernels
- trace: the first `prefill_flash` call of the trace's first file, rerun through the
  reference model with every decoder layer's self- and cross-attention input (after its
  LayerNorm) and output (before the out projection) captured, plus an FP32 truth per layer
  computed from those FP16 inputs and FP32 copies of the weights, and torch's layer-0
  cross K/V cache rows

Needs the GPU, so run it through the queue (scripts/gpu-test.sh does)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HEADS = 20
HEAD_DIM = 64
STATE = HEADS * HEAD_DIM
SELF_CAP = 448
CROSS_LEN = 1500
SCALE = HEAD_DIM**-0.5


def truth_attention(q, k_cache, v_cache, slots, lens, k_new, v_new):
    """FP32 attention as flash_attn_with_kvcache defines it, one row at a time

    q, k_new, v_new: [B, T, H, D]; caches [S, L, H, D]. Self-attention (k_new given):
    query t of row i sees the slot's first lens[slot] cached keys and new keys 0..=t.
    Cross-attention: every query sees the whole cache of the slot
    """
    B, T = q.shape[:2]
    out = torch.empty(B, T, HEADS, HEAD_DIM, dtype=torch.float32, device=q.device)
    for i in range(B):
        s = int(slots[i])
        if k_new is not None:
            n = int(lens[s])
            keys = torch.cat([k_cache[s, :n], k_new[i]]).float()
            vals = torch.cat([v_cache[s, :n], v_new[i]]).float()
            j = torch.arange(n + T, device=q.device)
            mask = j[None, :] <= (n + torch.arange(T, device=q.device))[:, None]
        else:
            keys = k_cache[s].float()
            vals = v_cache[s].float()
            mask = None
        scores = torch.einsum("thd,jhd->htj", q[i].float(), keys) * SCALE
        if mask is not None:
            scores = scores.masked_fill(~mask[None], float("-inf"))
        p = torch.softmax(scores, dim=-1)
        out[i] = torch.einsum("htj,jhd->thd", p, vals)
    return out


def synthetic(out, save, flash_attn_with_kvcache):
    """Random caches through flash-attn and the FP32 truth"""
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, device=dev, generator=g) * scale).half()

    # cached key counts before the step; with q_len new keys a row attends to up to
    # len + q_len keys, so 447 + 1 = 448 fills the cache
    ragged = [0, 1, 6, 7, 62, 63, 64, 65, 127, 128, 200, 300, 446, 447]
    specs = [
        # name, kind, rows (real, padding), q_len, real slots, lens
        ("self_decode_b52", "self", 46, 6, 1, 52, None),
        ("self_decode_b8", "self", 8, 0, 1, 8, [447, 0, 63, 64, 65, 1, 300, 446]),
        ("self_decode_b1", "self", 1, 0, 1, 4, [446]),
        ("self_decode_b1_short", "self", 1, 1, 1, 4, [5]),
        ("self_prefill_b5", "self", 5, 0, 7, 5, [0, 0, 0, 0, 0]),
        ("self_prefill_ragged", "self", 3, 1, 7, 3, [0, 100, 441]),
        ("cross_decode_b52", "cross", 47, 5, 1, 12, None),
        ("cross_decode_b2", "cross", 2, 0, 1, 3, None),
        ("cross_prefill_b5", "cross", 5, 0, 7, 5, None),
    ]

    cases = []
    for name, kind, real, pad, q_len, n_real, lens_spec in specs:
        rows = real + pad
        n_slots = n_real + 1
        trash = n_real
        cap = SELF_CAP if kind == "self" else CROSS_LEN
        perm = torch.randperm(n_real, device=dev, generator=g)
        if kind == "self":
            # real rows on distinct slots in a shuffled order
            slots = perm[:real]
        else:
            # cross-attention only reads, so rows may share a slot
            slots = perm[torch.arange(real, device=dev) % n_real]
        slots = torch.cat([slots, torch.full((pad,), trash, device=dev)]).int()

        lens = torch.zeros(n_slots, dtype=torch.int32, device=dev)
        if kind == "self":
            for i in range(real):
                lens[slots[i]] = (lens_spec or ragged)[i % len(lens_spec or ragged)]

        k_cache = randn(n_slots, cap, HEADS, HEAD_DIM)
        v_cache = randn(n_slots, cap, HEADS, HEAD_DIM)
        if kind == "self":
            pos = torch.arange(cap, device=dev)
            live = (pos[None, :] < lens[:, None].long())[:, :, None, None]
            k_cache = k_cache * live
            v_cache = v_cache * live
        q = randn(rows, q_len, HEADS, HEAD_DIM, scale=1.5)
        k_new = randn(rows, q_len, HEADS, HEAD_DIM) if kind == "self" else None
        v_new = randn(rows, q_len, HEADS, HEAD_DIM) if kind == "self" else None

        files = {
            "q": save(f"{name}.q", q),
            "k_cache": save(f"{name}.k_cache", k_cache),
            "v_cache": save(f"{name}.v_cache", v_cache),
            "slots": save(f"{name}.slots", slots),
        }
        truth = truth_attention(q, k_cache, v_cache, slots, lens, k_new, v_new)
        kc, vc = k_cache.clone(), v_cache.clone()
        if kind == "self":
            files["lens"] = save(f"{name}.lens", lens)
            files["k_new"] = save(f"{name}.k_new", k_new)
            files["v_new"] = save(f"{name}.v_new", v_new)
            flash = flash_attn_with_kvcache(
                q, kc, vc, k=k_new, v=v_new, cache_seqlens=lens[slots.long()],
                cache_batch_idx=slots, softmax_scale=SCALE, causal=True,
            )
        else:
            flash = flash_attn_with_kvcache(q, kc, vc, cache_batch_idx=slots, softmax_scale=SCALE, causal=False)
        files["flash"] = save(f"{name}.flash", flash)
        files["truth"] = save(f"{name}.truth", truth)
        cases.append({
            "name": name, "kind": kind, "rows": rows, "real_rows": real, "q_len": q_len,
            "slots": n_slots, "cap": cap, "files": files,
        })
    return cases


def trace(out, save, src, weights, trace_dir):
    """Per-layer attention inputs and outputs of one traced prefill"""
    from whisperdrz.model import WhisperDRZ

    stem = json.load(open(os.path.join(trace_dir, "summary.json")))["files"][0]["stem"]
    ev = None
    for line in open(os.path.join(trace_dir, stem, "events.jsonl")):
        e = json.loads(line)
        if e["ev"] == "prefill_flash":
            ev = e
            break
    assert ev is not None, f"no prefill_flash call in {trace_dir}/{stem}"
    a = ev["args"]
    tokens = torch.tensor(a["tokens"]["data"], device="cuda").view(a["tokens"]["shape"])
    slot_list = a["slot_indices"]["data"]
    slots = torch.tensor(slot_list, device="cuda")
    xa = torch.from_numpy(np.load(os.path.join(trace_dir, "blobs", a["xa"]["blob"] + ".npy"))).cuda()
    B, T = tokens.shape

    model = WhisperDRZ.from_pretrained_inf(weights).cuda()
    self_kv, cross_kv = model.create_flash_kv(max(slot_list) + 2, max_tokens=SELF_CAP)

    caps = {}

    def capture_input(mod, key):
        orig = mod.forward_flash

        def wrapped(x, *args, **kw):
            caps[key + ("x",)] = x.detach().clone()
            return orig(x, *args, **kw)

        mod.forward_flash = wrapped

    def capture_output(key):
        def hook(_mod, inp, _out):
            caps[key + ("o",)] = inp[0].detach().clone()

        return hook

    blocks = model.decoder.blocks
    for li, blk in enumerate(blocks):
        for part in ("attn", "cross_attn"):
            mod = getattr(blk, part)
            capture_input(mod, (part, li))
            mod.out.register_forward_hook(capture_output((part, li)))

    with torch.inference_mode():
        model.prefill_flash(tokens, xa, self_kv, cross_kv, slots)
        torch.cuda.synchronize()

        arrays = {}
        for part, key in (("attn", "self"), ("cross_attn", "cross")):
            xs, os_, truths = [], [], []
            for li, blk in enumerate(blocks):
                mod = getattr(blk, part)
                x = caps[(part, li, "x")]
                o = caps[(part, li, "o")]
                wq, bq = mod.query.weight.float(), mod.query.bias.float()
                wk = mod.key.weight.float()
                wv, bv = mod.value.weight.float(), mod.value.bias.float()
                src_kv = x.float() if key == "self" else xa.float()
                q = F.linear(x.float(), wq, bq).view(B, T, HEADS, HEAD_DIM)
                k = F.linear(src_kv, wk).view(B, -1, HEADS, HEAD_DIM)
                v = F.linear(src_kv, wv, bv).view(B, -1, HEADS, HEAD_DIM)
                if key == "self":
                    # a prefill starts from empty caches: attention over the T new tokens
                    zeros = torch.zeros(B, SELF_CAP, HEADS, HEAD_DIM, device="cuda")
                    lens = torch.zeros(B, dtype=torch.int32, device="cuda")
                    t = truth_attention(q, zeros, zeros, torch.arange(B), lens, k, v)
                else:
                    t = truth_attention(q, k, v, torch.arange(B), None, None, None)
                xs.append(x)
                os_.append(o.reshape(B, T, STATE))
                truths.append(t.reshape(B, T, STATE))
            arrays[f"{key}_x"] = save(f"trace.{key}_x", torch.stack(xs))
            arrays[f"{key}_flash"] = save(f"trace.{key}_flash", torch.stack(os_))
            arrays[f"{key}_truth"] = save(f"trace.{key}_truth", torch.stack(truths))
        arrays["xa"] = save("trace.xa", xa)
        arrays["cross_k0"] = save("trace.cross_k0", cross_kv[0].k_cache[slots].reshape(B, CROSS_LEN, STATE))
        arrays["cross_v0"] = save("trace.cross_v0", cross_kv[0].v_cache[slots].reshape(B, CROSS_LEN, STATE))

    return {
        "stem": stem, "seq": ev["seq"], "rows": B, "q_len": T, "slots": slot_list,
        "layers": len(blocks), "files": arrays,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--src", required=True, help="reference snapshot src dir, first on sys.path")
    p.add_argument("--weights", required=True)
    p.add_argument("--trace", help="trace set dir, such as .../ref-a783291/short3")
    args = p.parse_args()

    sys.path.insert(0, os.path.abspath(args.src))
    from flash_attn import __version__ as fa_version
    from flash_attn import flash_attn_with_kvcache

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.makedirs(args.out, exist_ok=True)

    def save(name, t):
        np.save(os.path.join(args.out, f"{name}.npy"), t.detach().cpu().numpy())
        return f"{name}.npy"

    manifest = {
        "torch": torch.__version__,
        "flash_attn": fa_version,
        "cases": synthetic(args.out, save, flash_attn_with_kvcache),
    }
    if args.trace:
        manifest["trace"] = trace(args.out, save, args.src, args.weights, args.trace)
    torch.cuda.synchronize()
    json.dump(manifest, open(os.path.join(args.out, "cases.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
