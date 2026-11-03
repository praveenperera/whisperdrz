"""Check that a traced run produced the same texts and segments as an untraced bench

usage: check_trace.py TRACE_DIR BENCH_TEXTS_JSON [--clip-s N]

Tracing must not perturb the pipeline; a mismatch means the hooks changed what
they record, and the fixtures can't serve as the oracle
"""

import json
import os
import sys


def main():
    trace_dir, bench_texts = sys.argv[1], sys.argv[2]
    clip = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--clip-s" else None
    bench = json.load(open(bench_texts))
    summary = json.load(open(os.path.join(trace_dir, "summary.json")))
    assert len(summary["files"]) == len(bench), (len(summary["files"]), len(bench))
    bad = 0
    for f in summary["files"]:
        stem = f["stem"]
        base = stem.removesuffix(f"-clip{clip}") if clip else stem
        res = json.load(open(os.path.join(trace_dir, stem, "result.json")))
        ref = bench[base + ".wav"]
        same = res["text"] == ref["text"] and res["segments"] == ref["segments"]
        bad += not same
        print(f"{stem}: {'same' if same else 'DIFF'} events={f['events']} sha={f['sha256'][:12]}")
    print(f"CHECK_TRACE {'OK' if bad == 0 else 'FAIL'} files={len(bench)} diff={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
