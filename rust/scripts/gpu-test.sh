#!/usr/bin/env bash
# Runs the host crate's GPU tests: torch reference GEMMs, then the cuda-feature tests
# against them and the checkpoint. Needs the GPU, so submit it through the queue; build
# first with `cargo test --release --features cuda --no-run` so the job spends no GPU
# time compiling
#
#   scripts/gpu-test.sh [extra cargo test args]
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
python="${WHISPERDRZ_PYTHON:-/home/praveen/code/whisperdrz/.venv/bin/python}"
ref="${WHISPERDRZ_GEMM_REF:-$HOME/.cache/whisperdrz/gemm-ref}"
export WHISPERDRZ_GEMM_REF="$ref"
export WHISPERDRZ_WEIGHTS="${WHISPERDRZ_WEIGHTS:-$(ls "$HOME"/.cache/huggingface/hub/models--fluxions--whisperdrz/snapshots/*/whisperdrz-large-v3.safetensors | head -1)}"
# cuBLAS 12.8.4 from the venv torch runs, so the native calls see the same library
export WHISPERDRZ_CUBLAS_DIR="${WHISPERDRZ_CUBLAS_DIR:-$("$python" -I -c 'import os, nvidia.cublas as c; print(os.path.join(c.__path__[0], "lib"))')}"

"$python" -I "$root/tools/gemm_ref.py" "$ref"
cd "$root"
cargo test --release --features cuda --no-fail-fast -p whisperdrz-native --test gpu_blas --test gpu_weights "$@" -- --nocapture --test-threads 1
