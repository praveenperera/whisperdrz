"""Torch FP16 GEMMs at whisperdrz's shapes, for the bitwise test of the native Blas calls

usage: gemm_ref.py OUT_DIR

Writes OUT_DIR/cases.json and one .npy per operand. Each case runs the same torch call
the model makes: `F.linear` with bias (addmm, cuBLASLt bias epilogue), `F.linear`
without bias (mm, cublasGemmEx) and the LM head `x @ W^T` on a contiguous transposed
copy of the token embedding. Needs the GPU, so run it through the queue
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROWS = [1, 7, 52, 54, 1500, 3000]
LINEARS = [("bias", 1280, 1280), ("bias", 1280, 5120), ("bias", 5120, 1280), ("nobias", 1280, 1280)]
VOCAB = 51866


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    dev = "cuda"
    cases = []

    def save(name, t):
        np.save(os.path.join(out, f"{name}.npy"), t.detach().cpu().numpy())
        return f"{name}.npy"

    weights = {}
    for kind, n_in, n_out in LINEARS:
        w = (torch.randn(n_out, n_in, device=dev) / n_in**0.5).half()
        b = (torch.randn(n_out, device=dev) * 0.1).half()
        weights[(n_in, n_out)] = (save(f"w{n_in}x{n_out}", w), w, save(f"b{n_out}", b), b)
    emb = (torch.randn(VOCAB, 1280, device=dev) * 0.02).half()
    emb_t = emb.t().contiguous()
    emb_t_file = save("emb_t", emb_t)

    for rows in ROWS:
        x_in = {n: torch.randn(rows, n, device=dev).half() for n in (1280, 5120)}
        x_file = {n: save(f"x{rows}x{n}", x) for n, x in x_in.items()}
        for kind, n_in, n_out in LINEARS:
            w_file, w, b_file, b = weights[(n_in, n_out)]
            x = x_in[n_in]
            y = F.linear(x, w, b) if kind == "bias" else F.linear(x, w)
            name = f"{kind}{rows}x{n_in}x{n_out}"
            case = {"kind": kind, "rows": rows, "n_in": n_in, "n_out": n_out, "x": x_file[n_in], "w": w_file, "y": save(name, y)}
            if kind == "bias":
                case["b"] = b_file
            cases.append(case)
        y = x_in[1280] @ emb_t
        cases.append({"kind": "lmhead", "rows": rows, "n_in": 1280, "n_out": VOCAB, "x": x_file[1280], "w": emb_t_file, "y": save(f"lmhead{rows}", y)})

    torch.cuda.synchronize()
    json.dump({"torch": torch.__version__, "cases": cases}, open(os.path.join(out, "cases.json"), "w"), indent=1)
    print(f"GEMM_REF wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
