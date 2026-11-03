#!/usr/bin/env bash
# Regenerates the committed PTX in kernels/ptx from the cuda-oxide kernel crate in
# kernels/, then its source manifest. One tier for now, compute_80: the attention kernels
# use FP16 mma.sync m16n8k16, and the Python path already needs sm_80 for flash-attn. The
# PTX stays at ISA 9.0 or below, which any driver from r580 can JIT
#
# cargo-oxide and its codegen backend come from the cuda-rust repository and commit that
# kernels/Cargo.lock pins, installed under a CARGO_HOME of their own per commit: the
# backend cache has one slot, so other projects at other commits would otherwise rebuild
# it under each other. Needs Linux, the crate's rust-toolchain.toml toolchain with its
# llvm-tools, libclang, and a CUDA toolkit (libdevice and ptxas). Codegen uses the
# toolchain's own llc, opt and llvm-link, never an llc-21 inherited from the environment,
# and stops when the toolchain's llc is missing. Every tool version, the cuda-oxide rev,
# the toolchain and the CUDA toolkit go to kernels/ptx/tools.txt, which the source
# manifest covers, so a rebuild with other tools shows up in the diff. Run it through the
# GPU queue on the GPU box
#
#   scripts/build-ptx.sh
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
crate="$root/kernels"
toolchain="$(sed -n 's/^channel = "\(.*\)"/\1/p' "$crate/rust-toolchain.toml")"
source="$(sed -n 's|^source = "git+\(https://github.com/[^/]*/cuda-rust.git\)?rev=[^#]*#\([0-9a-f]*\)"$|\1 \2|p' "$crate/Cargo.lock" | sort -u)"
repo="${source% *}"
rev="${source#* }"
if [[ ! "$rev" =~ ^[0-9a-f]{40}$ || "$repo" == "$source" ]]; then
  echo "expected one cuda-rust repository and commit in $crate/Cargo.lock, found: $source" >&2
  exit 1
fi

export CARGO_HOME="${WHISPERDRZ_OXIDE_HOME:-$HOME/.cache/whisperdrz/cargo-oxide-${rev:0:12}}"
export PATH="$CARGO_HOME/bin:$PATH"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
export LIBCLANG_PATH="${LIBCLANG_PATH:-/usr/lib/llvm-21/lib}"
export CUDA_OXIDE_LIBDEVICE="${CUDA_OXIDE_LIBDEVICE:-$cuda_home/nvvm/libdevice/libdevice.10.bc}"
# a cargo or rustup environment inherited from a parent build would pick another toolchain
unset RUSTUP_TOOLCHAIN CARGO RUSTC

# the toolchain's own llc, always: ~/code/cuda-rust-env.sh exports llc-21, and without the
# toolchain's llc cuda-oxide quietly takes an llc-23/22/21 from PATH, which moved the
# codegen of 31 of 66 sam3rs kernels with no source change
host="$(rustc "+$toolchain" -vV | sed -n 's/^host: //p')"
llc="$(rustc "+$toolchain" --print sysroot)/lib/rustlib/$host/bin/llc"
if [[ ! -x "$llc" ]]; then
  echo "no llc at $llc; run: rustup component add llvm-tools --toolchain $toolchain" >&2
  exit 1
fi
if [[ -n "${CUDA_OXIDE_LLC:-}" && "$CUDA_OXIDE_LLC" != "$llc" ]]; then
  echo "ignoring CUDA_OXIDE_LLC=$CUDA_OXIDE_LLC; using the toolchain's $llc" >&2
fi
export CUDA_OXIDE_LLC="$llc"
export CUDA_OXIDE_OPT="${llc%llc}opt"
export CUDA_OXIDE_LLVM_LINK="${llc%llc}llvm-link"
ptxas="$cuda_home/bin/ptxas"

stamp="$CARGO_HOME/cargo-oxide.rev"
if [[ "$(cat "$stamp" 2>/dev/null)" != "$repo $rev" ]]; then
  cargo "+$toolchain" install --locked --force --git "$repo" --rev "$rev" cargo-oxide
  echo "$repo $rev" > "$stamp"
fi

# every input that shapes the PTX besides the crate's own files; each value is a checked
# assignment, since a failure inside a command substitution in an echo would write an
# empty field and exit 0
llvm_version() {
  local version
  version="$("$1" --version | sed -n 's/^ *\(.*LLVM version .*\)$/\1/p')"
  if [[ -z "$version" ]]; then
    echo "no LLVM version from $1 --version" >&2
    return 1
  fi
  echo "$version"
}
llc_version="$(llvm_version "$CUDA_OXIDE_LLC")"
opt_version="$(llvm_version "$CUDA_OXIDE_OPT")"
link_version="$(llvm_version "$CUDA_OXIDE_LLVM_LINK")"
rustc_version="$(rustc "+$toolchain" -V)"
ptxas_version="$("$ptxas" --version | tail -1)"
libdevice_hash="$(sha256sum < "$CUDA_OXIDE_LIBDEVICE")"
libdevice_hash="${libdevice_hash%% *}"
[[ -n "$rustc_version" && -n "$ptxas_version" && -n "$libdevice_hash" ]]
printf 'cuda-oxide: %s %s\nrustc: %s\nllc: %s\nopt: %s\nllvm-link: %s\nlibdevice: %s\nptxas: %s\n' \
  "$repo" "$rev" "$rustc_version" "$llc_version" "$opt_version" "$link_version" "$libdevice_hash" \
  "$ptxas_version" > "$root/kernels/ptx/tools.txt"

cd "$crate"
for arch in 80; do
  out="$root/target/whisperdrz-kernels/sm$arch"
  mkdir -p "$out/ptx"
  # cargo-oxide only writes PTX when the crate itself recompiles
  cargo "+$toolchain" clean -q --target-dir "$out/cargo" -p whisperdrz-kernels
  CARGO_TARGET_DIR="$out/cargo" CUDA_OXIDE_PTX_DIR="$out/ptx" cargo "+$toolchain" oxide build --arch "sm_$arch"
  ptx="$out/ptx/whisperdrz_kernels.ptx"
  version="$(sed -n 's/^\.version \([0-9]*\.[0-9]*\)$/\1/p' "$ptx")"
  if ! awk -v v="$version" 'BEGIN { exit !(v != "" && v + 0 <= 9.0) }'; then
    echo "sm_$arch PTX has ISA version '$version', above the 9.0 that r580 drivers JIT" >&2
    exit 1
  fi
  # a call left in the PTX is a body that wasn't inlined, whose pointers then reach it as
  # generic ones: 25% slower in the sam3rs flash kernels
  if grep -q 'call\.uni\|^\s*call\b' "$ptx"; then
    echo "sm_$arch PTX has calls; make the callee a plain #[inline(always)] fn:" >&2
    grep -o 'call\.uni [A-Za-z0-9_]*' "$ptx" | sort -u >&2
    exit 1
  fi
  # registers, stack and spills per kernel for the target GPU, committed beside the PTX so
  # a codegen change shows in review
  "$ptxas" -arch=sm_120 -v "$ptx" -o "$out/sm120.cubin" 2> "$out/ptxas.txt"
  if grep -q 'bytes spill stores\|bytes stack frame' "$out/ptxas.txt" &&
    grep -E '[1-9][0-9]* bytes (spill stores|stack frame)' "$out/ptxas.txt" > /dev/null; then
    echo "sm_$arch PTX: ptxas reports stack or spills (allowed, but check):" >&2
    grep -B2 -E '[1-9][0-9]* bytes (spill stores|stack frame)' "$out/ptxas.txt" >&2
  fi
  sed 's/^ptxas info *: //' "$out/ptxas.txt" > "$root/kernels/ptx/ptxas.sm$arch.txt"
  cp "$ptx" "$root/kernels/ptx/whisperdrz.sm$arch.ptx"
done
"$root/scripts/check-ptx.sh" --write
echo "PTX rebuilt at cuda-oxide $rev with $rustc_version"
