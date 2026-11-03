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
# ECAPA2: weights and torch's per-layer truth, made once (the reference dump runs every
# recorded segment through torch, so it is reused while its manifest exists)
ecapa_ref="${WHISPERDRZ_ECAPA2_REF:-$HOME/.cache/whisperdrz/ecapa2/ref}"
export WHISPERDRZ_ECAPA2_REF="$ecapa_ref"
export WHISPERDRZ_ECAPA2_WEIGHTS="${WHISPERDRZ_ECAPA2_WEIGHTS:-$HOME/.cache/whisperdrz/ecapa2/ecapa2.safetensors}"
[[ -f "$WHISPERDRZ_ECAPA2_WEIGHTS" ]] || "$python" -I "$root/tools/ecapa2.py" export "$(dirname "$WHISPERDRZ_ECAPA2_WEIGHTS")"
[[ -f "$ecapa_ref/manifest.json" ]] || "$python" -I "$root/tools/ecapa2.py" ref "$ecapa_ref"
cd "$root"
# the Python reference is whisperdrz a783291's src/, extracted from git when missing
ref_src="${WHISPERDRZ_REF_SRC:-$HOME/.cache/whisperdrz/ref-a783291/src}"
if [[ ! -d "$ref_src/whisperdrz" ]]; then
  mkdir -p "$(dirname "$ref_src")"
  git -C "$(git -C "$root" rev-parse --show-toplevel)" archive a783291 src | tar -x -C "$(dirname "$ref_src")"
fi
traces="${WHISPERDRZ_FIXTURES:-$HOME/.cache/whisperdrz/fixtures/ref-a783291}"

# decoder attention: flash-attn and FP32 references on synthetic caches plus the layers of
# one traced prefill
attn_ref="${WHISPERDRZ_ATTN_REF:-$HOME/.cache/whisperdrz/attn-ref}"
"$python" -I "$root/tools/attn_ref.py" "$attn_ref" --src "$ref_src" --weights "$WHISPERDRZ_WEIGHTS" \
  --trace "$traces/short3"
export WHISPERDRZ_ATTN_REF="$attn_ref"

# encoder: torch per layer and end to end on recorded inputs (skips when its output is
# already there); the shared transformer kernels are checked against f64 in gpu_ops
export WHISPERDRZ_GPU=1
export WHISPERDRZ_ENCODER_REF="${WHISPERDRZ_ENCODER_REF:-$HOME/.cache/whisperdrz/encoder-ref}"
"$python" -I "$root/tools/encoder_layers.py" --src "$ref_src" --model "$WHISPERDRZ_WEIGHTS" "$WHISPERDRZ_ENCODER_REF"

# every GPU test binary runs even after one fails
cargo test --release --features cuda --no-fail-fast -p whisperdrz-native \
  --test gpu_blas --test gpu_weights --test gpu_decoder_attn --test gpu_ops --test gpu_encoder --test gpu_ecapa2 \
  "$@" -- --nocapture --test-threads 1
