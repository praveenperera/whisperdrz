"""Record the Python values the scheduler's unit tests compare against

- `sets`: CPython set iteration order over random add/discard histories. The
  transcriber refills slots from `list(empty_slots)[:n]`, so the Rust port
  emulates CPython's set table (`transcriber/pyset.rs`). Each case starts from a
  set comprehension over an ascending subset of slot ids, as
  `_run_batched_inference` builds `empty_slots`, then applies random adds and
  discards; `list(s)` is recorded after every op.
- `buckets`: `BucketedCUDAGraphs.capture_all(max_bucket)` bucket lists and the
  `_get_bucket` choice for every batch size up to 69.

Usage: python -I rust/tools/gen_transcriber_cases.py --src SNAPSHOT_SRC OUT.json
"""

import argparse
import json
import os
import random
import sys


def set_cases():
    rng = random.Random(1234)
    cases = []
    for _ in range(150):
        n = rng.choice([4, 8, 12, 20, 33, 52, 53, 64, 100])
        initial = sorted(rng.sample(range(n), rng.randint(0, n)))
        s = {i for i in initial}
        start = list(s)
        ops = []
        for _ in range(rng.randint(1, 80)):
            k = rng.randrange(n)
            op = rng.choice(["add", "discard"])
            getattr(s, op)(k)
            ops.append([op, k, list(s)])
        cases.append({"initial": initial, "start": start, "ops": ops})
    return cases


def bucket_cases():
    import whisperdrz.transcriber as T

    out = []
    for max_bucket in [1, 3, 8, 40, 52, 53, 54, 60]:
        g = T.BucketedCUDAGraphs.__new__(T.BucketedCUDAGraphs)
        # capture_all only needs the bucket list here, not the graphs
        g.warmup_and_capture = lambda b: None
        g.capture_all(max_bucket=max_bucket)
        out.append(
            {
                "max_bucket": max_bucket,
                "buckets": g.buckets,
                "choice": [T._get_bucket(b, g.buckets) for b in range(0, 70)],
            }
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="frozen whisperdrz src dir to import")
    p.add_argument("out")
    args = p.parse_args()
    sys.path.insert(0, os.path.abspath(args.src))
    data = {"python": sys.version, "sets": set_cases(), "buckets": bucket_cases()}
    with open(args.out, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))


if __name__ == "__main__":
    main()
