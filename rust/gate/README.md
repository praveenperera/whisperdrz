# Output gate

Reference texts and segments from the Python pipeline with the CUDA-graph padding fix
(`bench.py --pad-fix`, whisperdrz `a783291`, torch 2.8.0+cu128, flash-attn 2.8.3.post2,
RTX 5090). The native runtime must reproduce them byte for byte.

| Workload | Files | Audio | sha256 of `{"texts", "segments"}` |
| --- | --- | --- | --- |
| vox24, per-file `transcribe()` | 24 | 8049.8 s | `ba573d3660aa02c986dac43fb181e0078a161edff4490a1d8bcfa04e241730bf` |
| Earnings-21 4360674, one call | 1 | 3595.0 s | `e3418083a6d9fa6ac6cf71f0ecc4e2db91861d3b3fb4a76849c826ae21b485e8` |

The hash is `sha256(json.dumps({"texts": [...], "segments": [...]}, sort_keys=True, ensure_ascii=False))`
over the files in list order. Each was the same in two repeats in one process and in a
third process (gpu0 job `c39902b4`). The `*.ref.json` files map each wav's basename to
its `text` and `segments`.
