# CUDA Toolchain Report

## Selected Toolchain

- CUDA_HOME: `/home/elfo/Documents/gemma-4-12b/cuda_kernel_lab/toolchains/cuda-13.0-local/usr/local/cuda-13.0`
- nvcc path: `/home/elfo/Documents/gemma-4-12b/cuda_kernel_lab/toolchains/cuda-13.0-local/usr/local/cuda-13.0/bin/nvcc`
- TORCH_CUDA_ARCH_LIST: `12.0`

## Driver And GPU

- NVIDIA GeForce RTX 5080, 580.159.03, 12.0

## nvcc Version

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Aug_20_01:58:59_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0
```

## sm_120 Support

```text
sm_120
```

## PyTorch Runtime

```json
{
  "torch": "2.11.0+cu130",
  "torch_cuda": "13.0",
  "cuda_available": true,
  "gpu": "NVIDIA GeForce RTX 5080",
  "capability": [
    12,
    0
  ],
  "arch_list": [
    "sm_75",
    "sm_80",
    "sm_86",
    "sm_90",
    "sm_100",
    "sm_120"
  ]
}
```

## Smoke Tests

- Standalone native CUDA vector-add compiled with `-arch=sm_120`: pass.
- PyTorch `torch.utils.cpp_extension` vector-add compiled with `compute_120/sm_120`: pass.

## Notes

- Toolkit was extracted repo-locally from NVIDIA Ubuntu 24.04 CUDA 13.0 packages; no driver upgrade or system-wide CUDA symlink change was performed.
- Existing Ubuntu `nvidia-cuda-toolkit` 12.0 remains installed but is not used when sourcing `cuda_kernel_lab/env/cuda.sh`.
