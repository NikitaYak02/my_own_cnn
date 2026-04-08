#!/usr/bin/env bash
set -euo pipefail

configuration="${1:-Release}"
target="${2:-conv_bench}"

if ! command -v cmake >/dev/null 2>&1; then
  echo "Required command not found: cmake" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Required command not found: nvidia-smi" >&2
  exit 1
fi

gpu_info="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -n 1 | tr -d '\r')"
if [[ -z "${gpu_info}" ]]; then
  echo "Failed to detect NVIDIA GPU via nvidia-smi." >&2
  exit 1
fi

echo "Detected GPU: ${gpu_info}"
echo "Configuring preset: linux-auto-gcc"

cmake --preset linux-auto-gcc

echo "Building target '${target}' (${configuration})"
cmake --build build-linux-auto-gcc --target "${target}" --config "${configuration}" -j"$(nproc)"

echo "Build completed: build-linux-auto-gcc"
