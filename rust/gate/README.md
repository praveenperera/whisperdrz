# Output gate

Reference texts and segments from the Python pipeline with the CUDA-graph padding fix
(`bench.py --pad-fix`, whisperdrz `a783291`, torch 2.8.0+cu128, flash-attn 2.8.3.post2,
RTX 5090). The native runtime must stay inside the noise band of these outputs, checked by
`gate.py`; byte identity is reported but not required.

| Workload | Files | Audio | sha256 of `{"texts", "segments"}` |
| --- | --- | --- | --- |
| vox24, per-file `transcribe()` | 24 | 8049.8 s | `ba573d3660aa02c986dac43fb181e0078a161edff4490a1d8bcfa04e241730bf` |
| Earnings-21 4360674, one call | 1 | 3595.0 s | `e3418083a6d9fa6ac6cf71f0ecc4e2db91861d3b3fb4a76849c826ae21b485e8` |

The hash is `sha256(json.dumps({"texts": [...], "segments": [...]}, sort_keys=True, ensure_ascii=False))`
over the files in list order. Each was the same in two repeats in one process and in a
third process (gpu0 job `c39902b4`). The `*.ref.json` files map each wav's basename to
its `text` and `segments`.

## Noise-band gate

    /home/praveen/code/whisperdrz/.venv/bin/python gate.py {vox24|e21} NEW_TEXTS_JSON

The reference itself isn't stable under changes that are exact in real arithmetic. Rerun
with an f64 mel, the memory-efficient encoder SDPA, or cross-attention without split-KV,
it changed words or speaker turns on 10 to 17 of the 24 vox24 files and on e21, while DER
stayed within 1.3 points (gpu0 jobs `4bdc49e2` and `939728ce`). So `gate.py` passes a run
when, against the reference, its word edits and cpWER-style edits (timestamps stripped)
and its DER against the RTTMs (0.25 s collar, overlap scored) stay inside the band those
reruns measured:

| Workload | Reference DER | DER limit | Word edits | cpWER-style edits |
| --- | --- | --- | --- | --- |
| vox24 | 21.98 | 23.23 | 1.0% | 6.0% |
| e21 | 27.47 | 27.97 | 0.1% | 0.3% |

Word timings are gated too, since the text metrics strip timestamps and DER reads only
segments. Words are aligned over the timestamp-free word sequence; a boundary's error is
taken where both runs time the same aligned word. The timed-word share may fall at most
0.5 points below the reference's (every rerun stayed within 0.13), and the mean boundary
error and the share of boundaries moved over 0.2 s may reach 1.5x the worst rerun:

| Workload | Timed words (reference) | Mean boundary error limit | Moved > 0.2 s limit |
| --- | --- | --- | --- |
| vox24 | 96.83% | 0.313 s (worst rerun 0.209) | 4.39% (worst rerun 2.93) |
| e21 | 98.75% | 0.266 s (worst rerun 0.178) | 4.44% (worst rerun 2.96) |

Dropping every interior timestamp of a line, or shifting every timestamp by 0.5 s, fails.

It first fails on malformed output (missing or reordered files, empty text, no segments,
non-finite times), and prints the first divergence of each differing file to localize
bugs. Every perturbation arm passes, and the reference passes against itself
