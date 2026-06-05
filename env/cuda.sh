#!/usr/bin/env bash
# Project-local CUDA 13.0 toolchain extracted from NVIDIA Ubuntu 24.04 packages.
# Source this file before native CUDA compilation:
#   source cuda_kernel_lab/env/cuda.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export CUDA_HOME="$REPO_ROOT/cuda_kernel_lab/toolchains/cuda-13.0-local/usr/local/cuda-13.0"
export CUDA_PATH="$CUDA_HOME"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/nvvm/lib64:${LD_LIBRARY_PATH:-}"

