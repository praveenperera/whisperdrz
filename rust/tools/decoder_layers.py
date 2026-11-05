"""Per-layer torch references for the native text decoder and its DTW forward

usage: decoder_layers.py --src SNAPSHOT_SRC --model CKPT --trace TRACE_SET_DIR OUT_DIR

Decode (`tests/gpu_decoder.rs`): the first `prefill_flash` call of the trace set's first
file (5 windows on slots 0..4, 7 prompt tokens each), then five decode steps on caches
with a trash slot (5) as the padding fix sets them up, one of every bucket from 5 rows
down to 1:

  step1  4 rows on slots [3, 0, 4, 1] at position 7, plus a padding row (token 0,
         position 0, trash slot)
  step2  3 rows on slots [2, 0, 4] at positions [7, 8, 8] (ragged cache lengths), plus a
         padding row
  step3  2 rows on slots [1, 3] at position 8, plus a padding row
  step4  1 row on slot 2 at position 8, plus a padding row
  step5  1 row on slot 1 at position 9, no padding row

Each step's tokens are torch FP16's argmax at the row's last position, fixed once so
every run below sees the same inputs. Per phase (`prefill`, `step1`..`step5`) and layer
(`embed`, `block00`..`block31`, `ln`, `logits`):

  <phase>/f16_<name>.npy     torch FP16 as the transcriber runs it (flash-attn caches)
  <phase>/f16qkv_<name>.npy  FP16 with the self-attention q, k, v as one packed GEMM with a
                             zero key bias (the native layout), exact in real arithmetic
  <phase>/f16sdpa_<name>.npy FP16 with the model's SDPA fallback for flash_attn_with_kvcache,
                             also exact in real arithmetic. Its unmasked cross-attention
                             takes SDPA's flash backend, the same FA2 as flash-attn
  <phase>/f16eff_<name>.npy  FP16 with the SDPA fallback on the memory-efficient backend for
                             both attentions: the one variant that moves cross-attention's
                             summation order, the only freedom a position-0 row has
  <phase>/f32_<name>.npy     FP32 on the same FP16 weights, SDPA math backend, TF32 off:
                             the chained truth
  <phase>/iso_<name>.npy     FP32 layer <name> on torch FP16's previous layer output and
                             torch FP16's caches before the phase: the isolated truth
  <phase>/isoeff_<name>.npy  the f16eff variant's block <name> on the same input and
                             caches: the isolated noise yardstick (blocks only)
  sub/<phase>/bNN_<op>.npy    torch FP16's block NN rebuilt one piece at a time from its
                             recorded input (h1 = attn_ln(x), a = self-attention(h1), x1 =
                             x + a, h2 = cross_attn_ln(x1), c = cross-attention(h2), x2 =
                             x1 + c, h3 = mlp_ln(x2), m = mlp(h3)); the manifest's
                             `sub_differs` counts values of x2 + m off the recorded block
  sub/<phase>/bNN_<op>32.npy  the FP32 truth of each piece (h1, a, h2, c, h3, m) on torch
                             FP16's input to it
  <phase>/self_k.npy, self_v.npy, lens.npy   torch FP16's self-attention caches (positions
                             0..9) and lengths before the phase, `[32, 6, 10, 1280]`
  <phase>/inputs.json        tokens, positions, slots

DTW (`tests/gpu_dtw.rs`): every recorded `dtw_decoder_forward` call of every file, rerun
in FP16 (checked bit for bit against the recorded capture) and in FP32 (the truth), plus
the errors against the truth of two FP16 variants exact in real arithmetic (packed QKV,
memory-efficient SDPA) in the manifest:

  dtw/<stem>-<seq>.scores32.npy    FP32 capture `[B, 10, T, 1500]` before the prep
  dtw/<stem>-<seq>.matrices32.npy  `_dtw_prep_batch` of it
  dtw/capture-*.npy                the FP16 query and key projections the hooks saw on one
                                   small call, for a bitwise check of the capture GEMM

Needs the GPU, so run it through the queue (`scripts/gpu-test.sh` does); skips when the
manifest of this version is already there
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

VERSION = 5
N_SLOTS = 5
TRASH = N_SLOTS
SELF_CAP = 448
KEEP = 10
STEP1_SLOTS = [3, 0, 4, 1]
STEP2_SLOTS = [2, 0, 4]
STEP3_SLOTS = [1, 3]
STEP4_SLOTS = [2]
STEP5_SLOTS = [1]
PHASES = ("prefill", "step1", "step2", "step3", "step4", "step5")
# the capture's bitwise check: a call with one short row keeps the dump small
CAPTURE_CALL = ("nctdh", 365)


def load_events(trace, stem):
    with open(os.path.join(trace, stem, "events.jsonl"), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def plain(v):
    """A recorded list argument, logged either as JSON or as its repr"""
    return json.loads(v) if isinstance(v, str) else v


def array(trace, v):
    if "data" in v:
        return np.array(v["data"], dtype=v["__a"]).reshape(v["shape"])
    assert v.get("saved"), f"blob {v.get('blob')} not saved"
    # blobs are stored by content, so one first saved by another call with the same bytes
    # (a slot's [1500, 1280] encoder output, say) carries that call's shape
    a = np.load(os.path.join(trace, "blobs", v["blob"] + ".npy"))
    assert a.size == int(np.prod(v["shape"])), (a.shape, v["shape"])
    return a.reshape(v["shape"])


class Capture:
    """Records the embedding, every block's output and the final LayerNorm of
    `TextDecoder.forward_flash` by wrapping the blocks' `forward_flash`"""

    def __init__(self, decoder):
        self.out = {}
        self.blocks = list(decoder.blocks)
        for i, blk in enumerate(self.blocks):
            orig = blk.forward_flash

            def wrapped(x, *a, _i=i, _orig=orig, **kw):
                if _i == 0:
                    self.out["embed"] = x.detach().clone()
                y = _orig(x, *a, **kw)
                self.out[f"block{_i:02d}"] = y.detach().clone()
                return y

            blk.forward_flash = wrapped
        self.hook = decoder.ln.register_forward_hook(
            lambda m, a, y: self.out.__setitem__("ln", y.detach().clone())
        )

    def close(self):
        for blk in self.blocks:
            del blk.forward_flash
        self.hook.remove()


def packed_qkv(M):
    """Patches self-attention's forward_flash to one packed QKV GEMM; returns an undo"""
    orig = M.MultiHeadAttention.forward_flash

    def forward_flash(self, x, kv_cache, is_cross_attn=False, slot_indices=None):
        if is_cross_attn:
            return orig(self, x, kv_cache, is_cross_attn, slot_indices)
        B, T, _ = x.shape
        hd = self.n_state // self.n_head
        w = torch.cat([self.query.weight, self.key.weight, self.value.weight])
        b = torch.cat([self.query.bias, torch.zeros_like(self.query.bias), self.value.bias])
        qkv = F.linear(x, w, b)
        q, k, v = (t.reshape(B, T, self.n_head, hd) for t in qkv.split(self.n_state, dim=-1))
        lens = kv_cache.seq_lens if slot_indices is None else kv_cache.seq_lens[slot_indices]
        idx = slot_indices.to(torch.int32) if slot_indices is not None else None
        out = M.flash_attn_with_kvcache(
            q, kv_cache.k_cache, kv_cache.v_cache, k=k, v=v, cache_seqlens=lens,
            cache_batch_idx=idx, softmax_scale=hd**-0.5, causal=self.causal,
        )
        return self.out(out.reshape(B, T, self.n_state))

    M.MultiHeadAttention.forward_flash = forward_flash
    return lambda: setattr(M.MultiHeadAttention, "forward_flash", orig)


def packed_qkv_full(M):
    """Patches the uncached self-attention (the DTW forward's) to one packed QKV GEMM with a
    zero key bias; cross-attention, which the capture hooks, is unchanged. Returns an undo"""
    orig = M.MultiHeadAttention.forward

    def forward(self, x, xa=None, mask=None):
        if xa is not None:
            return orig(self, x, xa, mask)
        w = torch.cat([self.query.weight, self.key.weight, self.value.weight])
        b = torch.cat([self.query.bias, torch.zeros_like(self.query.bias), self.value.bias])
        q, k, v = F.linear(x, w, b).split(self.n_state, dim=-1)
        return self.qkv_attention(q, k, v, mask)

    M.MultiHeadAttention.forward = forward
    return lambda: setattr(M.MultiHeadAttention, "forward", orig)


def err_vs(a, truth):
    """max-abs and mean-abs of `a` against `truth`, in f64"""
    d = (torch.as_tensor(a).cuda().double() - torch.as_tensor(truth).cuda().double()).abs()
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item()}


def sdpa_fallback(M):
    """Routes flash_attn_with_kvcache to the model's SDPA fallback; returns an undo. The
    fallback returns a permuted view that forward_flash's .view rejects, so it is made
    contiguous, which changes no value"""
    orig = M.flash_attn_with_kvcache
    M.flash_attn_with_kvcache = lambda *a, **kw: M._sdpa_attn_with_kvcache(*a, **kw).contiguous()
    return lambda: setattr(M, "flash_attn_with_kvcache", orig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("out")
    args = ap.parse_args()

    manifest_path = os.path.join(args.out, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            if json.load(fh).get("version") == VERSION:
                print(f"decoder references already in {args.out}")
                return

    sys.path.insert(0, os.path.abspath(args.src))
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import whisperdrz.model as M
    from whisperdrz.dtw_align import CrossAttnCapture, _dtw_prep_batch, make_head_mask

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def save(rel, t):
        path = os.path.join(args.out, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
        np.save(path, t)
        return rel

    m16 = M.WhisperDRZ.from_pretrained_inf(args.model).cuda()
    m16.decoder.cache_embedding_weight(torch.float16)
    m32 = M.WhisperDRZ.from_pretrained(args.model).cuda().eval()
    m32.decoder.cache_embedding_weight(torch.float32)
    # the same FP16 timing table, widened, so both models see the same embedding inputs
    m32.decoder.timing_embedding = m16.decoder.timing_embedding.float()
    del m16.encoder, m32.encoder
    torch.cuda.empty_cache()

    summary = json.load(open(os.path.join(args.trace, "summary.json")))
    stem = summary["files"][0]["stem"]
    events = load_events(args.trace, stem)
    ev = next(e for e in events if e["ev"] == "prefill_flash")
    a = ev["args"]
    tokens0 = torch.from_numpy(array(args.trace, a["tokens"])).cuda()
    slots0 = torch.from_numpy(array(args.trace, a["slot_indices"])).cuda()
    assert slots0.tolist() == list(range(N_SLOTS)), slots0.tolist()
    xa16 = torch.from_numpy(array(args.trace, a["xa"])).cuda()

    def run(model, kind, tokens_of=None):
        """Prefill and the steps; returns per-phase captures, logits and inputs"""
        dt = model.dtype
        self_kv, cross_kv = model.create_flash_kv(N_SLOTS + 1, max_tokens=SELF_CAP)
        xa = xa16.to(dt)
        phases = {}

        def phase(name, fn, slots, tokens, positions):
            state = {
                "self_k": torch.stack([kv.k_cache[:, :KEEP] for kv in self_kv]).clone(),
                "self_v": torch.stack([kv.v_cache[:, :KEEP] for kv in self_kv]).clone(),
                "lens": self_kv[0].seq_lens.clone(),
            }
            cap = Capture(model.decoder)
            try:
                logits = fn()
            finally:
                cap.close()
            cap.out["logits"] = logits.reshape(-1, logits.shape[-1])
            phases[name] = {
                "out": cap.out, "state": state, "slots": slots, "tokens": tokens,
                "positions": positions,
            }
            return logits

        def step(name, real_slots, toks, poss, pad=True):
            for kv in self_kv:
                kv.seq_lens[TRASH] = 0
            # the padding row: token 0 at position 0 on the trash slot
            extra = 1 if pad else 0
            slots = real_slots + [TRASH] * extra
            toks, poss = toks + [0] * extra, poss + [0] * extra
            tok = torch.tensor(toks, device="cuda").view(-1, 1)
            pos = torch.tensor(poss, device="cuda")
            sl = torch.tensor(slots, device="cuda")
            return phase(
                name,
                lambda: model.decoder.forward_flash(tok, self_kv, cross_kv, pos, slot_indices=sl),
                slots, toks, poss,
            )

        def ctx():
            if kind == "f32":
                return sdpa_kernel([SDPBackend.MATH])
            if kind == "f16eff":
                return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
            return torch.inference_mode()

        undo_sdpa = sdpa_fallback(M) if kind in ("f32", "f16sdpa", "f16eff") else (lambda: None)
        undo = packed_qkv(M) if kind == "f16qkv" else (lambda: None)
        try:
            with torch.inference_mode(), ctx():
                T = tokens0.shape[1]
                lp = phase(
                    "prefill",
                    lambda: model.prefill_flash(tokens0, xa, self_kv, cross_kv, slots0),
                    list(range(N_SLOTS)), tokens0.reshape(-1).tolist(),
                    list(range(T)) * N_SLOTS,
                )
                # the step tokens come from torch FP16's own run, then stay fixed
                if tokens_of is None:
                    first = lp[:, -1].argmax(-1).tolist()
                    toks1 = [first[s] for s in STEP1_SLOTS]
                else:
                    toks1 = tokens_of["step1"]
                l1 = step("step1", STEP1_SLOTS, toks1, [T] * len(STEP1_SLOTS))
                if tokens_of is None:
                    nxt = dict(zip(STEP1_SLOTS, l1[:, -1].argmax(-1).tolist()))
                    toks2 = [first[2], nxt[0], nxt[4]]
                else:
                    toks2 = tokens_of["step2"]
                l2 = step("step2", STEP2_SLOTS, toks2, [T, T + 1, T + 1])
                if tokens_of is None:
                    toks3 = [nxt[1], nxt[3]]
                else:
                    toks3 = tokens_of["step3"]
                l3 = step("step3", STEP3_SLOTS, toks3, [T + 1, T + 1])
                if tokens_of is None:
                    toks4 = [l2[0, -1].argmax(-1).item()]
                else:
                    toks4 = tokens_of["step4"]
                step("step4", STEP4_SLOTS, toks4, [T + 1])
                if tokens_of is None:
                    toks5 = [l3[0, -1].argmax(-1).item()]
                else:
                    toks5 = tokens_of["step5"]
                step("step5", STEP5_SLOTS, toks5, [T + 2], pad=False)
        finally:
            undo()
            undo_sdpa()
        return phases, self_kv, cross_kv

    # the FP16 table the DTW forward adds, for a bitwise check of the native one
    save("timing.npy", m16.decoder.timing_embedding)

    names = ["embed"] + [f"block{i:02d}" for i in range(32)] + ["ln", "logits"]
    runs = {}
    runs["f16"], _, cross16 = run(m16, "f16")
    recorded = array(args.trace, ev["ret"])
    got = runs["f16"]["prefill"]["out"]["logits"].reshape(recorded.shape).cpu().numpy()
    prefill_differs = int((got.view(np.uint16) != recorded.view(np.uint16)).sum())
    print(f"prefill logits rerun vs recorded: {prefill_differs} values differ")
    # the real rows' tokens, which lead the padding row's
    fixed = {
        p: runs["f16"][p]["tokens"][: sum(s != TRASH for s in runs["f16"][p]["slots"])]
        for p in PHASES[1:]
    }
    for kind, model in (("f16qkv", m16), ("f16sdpa", m16), ("f16eff", m16), ("f32", m32)):
        runs[kind], _, _ = run(model, kind, fixed)

    manifest = {
        "version": VERSION, "stem": stem, "seq": ev["seq"], "names": names, "phases": {},
        "prefill_differs": prefill_differs, "timing": "timing.npy",
    }
    for p in PHASES:
        ref = runs["f16"][p]
        for kind in runs:
            assert runs[kind][p]["tokens"] == ref["tokens"], (kind, p)
            for n in names:
                save(f"{p}/{kind}_{n}.npy", runs[kind][p]["out"][n])
        st = ref["state"]
        save(f"{p}/self_k.npy", st["self_k"])
        save(f"{p}/self_v.npy", st["self_v"])
        save(f"{p}/lens.npy", st["lens"])
        manifest["phases"][p] = {
            "tokens": ref["tokens"], "positions": ref["positions"], "slots": ref["slots"],
            "rows": len(ref["slots"]), "q_len": len(ref["tokens"]) // len(ref["slots"]),
        }

    # sub-ops of torch FP16's blocks on copies of its caches before the phase, on the
    # flash-attn path the run took
    sub_in = {}
    manifest["sub_differs"] = {}
    with torch.inference_mode():
        for p in PHASES:
            ref = runs["f16"][p]
            st = ref["state"]
            sl = torch.tensor(ref["slots"], device="cuda")
            differs = []
            for i in range(32):
                blk = m16.decoder.blocks[i]
                x = ref["out"]["embed" if i == 0 else f"block{i - 1:02d}"]
                kv = M.FlashKVCache(N_SLOTS + 1, SELF_CAP, 20, 64, dtype=torch.float16, device="cuda")
                kv.k_cache[:, :KEEP] = st["self_k"][i]
                kv.v_cache[:, :KEEP] = st["self_v"][i]
                kv.seq_lens.copy_(st["lens"])
                if p == "prefill":
                    kv.seq_lens[sl] = 0
                t = {"x": x}
                t["h1"] = blk.attn_ln(x)
                t["a"] = blk.attn.forward_flash(t["h1"], kv, is_cross_attn=False, slot_indices=sl)
                t["x1"] = x + t["a"]
                t["h2"] = blk.cross_attn_ln(t["x1"])
                t["c"] = blk.cross_attn.forward_flash(t["h2"], cross16[i], is_cross_attn=True, slot_indices=sl)
                t["x2"] = t["x1"] + t["c"]
                t["h3"] = blk.mlp_ln(t["x2"])
                t["m"] = blk.mlp(t["h3"])
                differs.append(int((t["x2"] + t["m"] != ref["out"][f"block{i:02d}"]).sum().item()))
                for k in ("h1", "a", "x1", "h2", "c", "x2", "h3", "m"):
                    save(f"sub/{p}/b{i:02d}_{k}.npy", t[k])
                sub_in[(p, i)] = t
            manifest["sub_differs"][p] = differs
            print(f"sub-ops {p}: rebuilt blocks differ from the recorded ones in {differs}")

    # isolated truths: FP32 layer i on torch FP16's input to it, against FP32 copies of
    # torch FP16's caches before the phase (cross K/V recomputed in FP32 from xa)
    undo_sdpa = sdpa_fallback(M)
    dec32 = m32.decoder
    with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
        cross32 = []
        for blk in dec32.blocks:
            kv = M.FlashKVCache(N_SLOTS + 1, 1500, 20, 64, dtype=torch.float32, device="cuda")
            kv.set_cross_attention(
                blk.cross_attn.key(xa16.float()).view(N_SLOTS, -1, 20, 64),
                blk.cross_attn.value(xa16.float()).view(N_SLOTS, -1, 20, 64),
                slots0,
            )
            cross32.append(kv)
        for p in PHASES:
            ref = runs["f16"][p]
            st = ref["state"]
            sl = torch.tensor(ref["slots"], device="cuda")
            rows, q_len = len(ref["slots"]), len(ref["tokens"]) // len(ref["slots"])
            for n in names:
                if n == "embed":
                    iso = runs["f32"][p]["out"]["embed"]
                elif n.startswith("block"):
                    i = int(n[5:])
                    prev = ref["out"]["embed" if i == 0 else f"block{i - 1:02d}"]
                    kv = M.FlashKVCache(N_SLOTS + 1, SELF_CAP, 20, 64, dtype=torch.float32, device="cuda")
                    kv.k_cache[:, :KEEP] = st["self_k"][i].float()
                    kv.v_cache[:, :KEEP] = st["self_v"][i].float()
                    kv.seq_lens.copy_(st["lens"])
                    if p == "prefill":
                        kv.seq_lens[sl] = 0
                    iso = dec32.blocks[i].forward_flash(prev.float(), kv, cross32[i], slot_indices=sl)
                elif n == "ln":
                    iso = dec32.ln(ref["out"]["block31"].float())
                else:
                    iso = ref["out"]["ln"].float() @ dec32._embedding_weight_T
                    iso = iso.reshape(rows * q_len, -1)
                save(f"{p}/iso_{n}.npy", iso)
            for i in range(32):
                blk = dec32.blocks[i]
                t = sub_in[(p, i)]
                kv = M.FlashKVCache(N_SLOTS + 1, SELF_CAP, 20, 64, dtype=torch.float32, device="cuda")
                kv.k_cache[:, :KEEP] = st["self_k"][i].float()
                kv.v_cache[:, :KEEP] = st["self_v"][i].float()
                kv.seq_lens.copy_(st["lens"])
                if p == "prefill":
                    kv.seq_lens[sl] = 0
                truth = {
                    "h1": blk.attn_ln(t["x"].float()),
                    "a": blk.attn.forward_flash(t["h1"].float(), kv, is_cross_attn=False, slot_indices=sl),
                    "h2": blk.cross_attn_ln(t["x1"].float()),
                    "c": blk.cross_attn.forward_flash(t["h2"].float(), cross32[i], is_cross_attn=True, slot_indices=sl),
                    "h3": blk.mlp_ln(t["x2"].float()),
                    "m": blk.mlp(t["h3"].float()),
                }
                for k, v in truth.items():
                    save(f"sub/{p}/b{i:02d}_{k}32.npy", v)
    # the same isolated layers in torch FP16 on the memory-efficient backend, against
    # copies of its caches (decode steps write into the self cache)
    with torch.inference_mode(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        for p in PHASES:
            ref = runs["f16"][p]
            st = ref["state"]
            sl = torch.tensor(ref["slots"], device="cuda")
            for i in range(32):
                prev = ref["out"]["embed" if i == 0 else f"block{i - 1:02d}"]
                kv = M.FlashKVCache(N_SLOTS + 1, SELF_CAP, 20, 64, dtype=torch.float16, device="cuda")
                kv.k_cache[:, :KEEP] = st["self_k"][i]
                kv.v_cache[:, :KEEP] = st["self_v"][i]
                kv.seq_lens.copy_(st["lens"])
                if p == "prefill":
                    kv.seq_lens[sl] = 0
                out = m16.decoder.blocks[i].forward_flash(prev, kv, cross16[i], slot_indices=sl)
                save(f"{p}/isoeff_block{i:02d}.npy", out)
    undo_sdpa()

    # DTW: rerun every recorded decoder forward with the capture hooks
    head_mask = make_head_mask(m16).cuda()
    layers = {int(l) for l in head_mask.nonzero()[:, 0].tolist()}
    dtw_cases = []
    for f in summary["files"]:
        evs = load_events(args.trace, f["stem"])
        fwds = [e for e in evs if e["ev"] == "dtw_decoder_forward"]
        preps = [e for e in evs if e["ev"] == "_dtw_prep_batch"]
        for fwd in fwds:
            prep = min((p for p in preps if p["seq"] > fwd["seq"]), key=lambda p: p["seq"])
            x = torch.from_numpy(array(args.trace, fwd["args"]["x"])).cuda()
            xa = torch.from_numpy(array(args.trace, fwd["args"]["xa"])).cuda()
            recorded = array(args.trace, prep["args"]["full_weights"])
            lens, caps = (plain(prep["args"][k]) for k in ("actual_lens", "audio_caps"))

            def capture(model, backends=None):
                with CrossAttnCapture(model, layers=layers) as cap:
                    with torch.inference_mode():
                        if backends is None:
                            model.decoder(x, xa.to(model.dtype))
                        else:
                            with sdpa_kernel(backends):
                                model.decoder(x, xa.to(model.dtype))
                    w = cap.compute_attention(head_mask)
                    if w.dim() == 3:
                        w = w.unsqueeze(0)
                    return w, cap

            w16, cap16 = capture(m16)
            w16_flash, _ = capture(m16, [SDPBackend.FLASH_ATTENTION])
            w32, _ = capture(m32, [SDPBackend.MATH])
            m32_mat = _dtw_prep_batch(w32, lens, caps).copy()
            # torch FP16 variants exact in real arithmetic, the bar for chained max-abs: the
            # native packed QKV layout, and memory-efficient SDPA for both attentions
            undo = packed_qkv_full(M)
            try:
                w_qkv, _ = capture(m16)
            finally:
                undo()
            w_eff, _ = capture(m16, [SDPBackend.EFFICIENT_ATTENTION])
            variants = {}
            for vname, wv in (("f16qkv", w_qkv), ("f16eff", w_eff)):
                mv = _dtw_prep_batch(wv, lens, caps).copy()
                variants[vname] = {"scores": err_vs(wv, w32), "matrices": err_vs(mv, m32_mat)}
            name = f"{f['stem']}-{fwd['seq']}"
            case = {
                "stem": f["stem"], "forward_seq": fwd["seq"], "prep_seq": prep["seq"],
                "scores32": save(f"dtw/{name}.scores32.npy", w32),
                "matrices32": save(f"dtw/{name}.matrices32.npy", m32_mat),
                "rerun_differs": int((w16.cpu().numpy() != recorded).sum()),
                "flash_differs": int((w16_flash != w16).sum().item()),
                "variants": variants,
            }
            if (f["stem"], fwd["seq"]) == CAPTURE_CALL:
                case["capture"] = {
                    "q": {str(l): save(f"dtw/capture-q{l}.npy", cap16.q_out[l]) for l in sorted(layers)},
                    "k": {str(l): save(f"dtw/capture-k{l}.npy", cap16.k_out[l]) for l in sorted(layers)},
                    "scores16": save("dtw/capture-scores16.npy", w16),
                }
            print(f"dtw {name}: B={x.shape[0]} T={x.shape[1]} rerun vs recorded differ "
                  f"{case['rerun_differs']}, flash-only vs default differ {case['flash_differs']}")
            dtw_cases.append(case)
    manifest["dtw"] = dtw_cases
    assert any("capture" in c for c in dtw_cases), f"no DTW call {CAPTURE_CALL}"

    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"wrote decoder references to {args.out}")


if __name__ == "__main__":
    main()
