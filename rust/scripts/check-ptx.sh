#!/usr/bin/env bash
# Fails when the committed PTX is stale: kernels/ptx/sources.sha256 records the hashes of
# the kernel crate's sources, manifest, lockfile (the cuda-oxide rev) and toolchain pin,
# the build script, the tools it built with (kernels/ptx/tools.txt: cuda-oxide rev, rustc,
# LLVM, libdevice, the CUDA toolkit's ptxas) and the PTX it produced, so editing any of
# them without rerunning scripts/build-ptx.sh changes a hash. Needs no CUDA toolkit or
# cuda-oxide, so CI can run it anywhere
#
#   scripts/check-ptx.sh            # check
#   scripts/check-ptx.sh --write    # record the current files (build-ptx.sh does this)
set -euo pipefail

cd "$(dirname "$0")/.."
manifest=kernels/ptx/sources.sha256
crate=kernels
mapfile -t sources < <(find "$crate/src" -name '*.rs' | LC_ALL=C sort)
files=("${sources[@]}" "$crate/Cargo.toml" "$crate/Cargo.lock" "$crate/rust-toolchain.toml"
  scripts/build-ptx.sh kernels/ptx/tools.txt kernels/ptx/*.ptx kernels/ptx/ptxas.*.txt)

if [[ "${1:-}" == "--write" ]]; then
  sha256sum "${files[@]}" > "$manifest"
  exit 0
fi

if ! diff <(sha256sum "${files[@]}") "$manifest"; then
  echo "kernels/ptx is stale: run scripts/build-ptx.sh through the GPU queue and commit the result" >&2
  exit 1
fi
echo "kernels/ptx matches its sources"
