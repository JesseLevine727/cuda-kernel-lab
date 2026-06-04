# Environment Report

## Local GPU Runtime

- Python/platform: `3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]` / `Linux-6.17.0-1023-oem-x86_64-with-glibc2.39`
- PyTorch: `2.11.0+cu130`
- PyTorch CUDA: `13.0`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 5080`
- GPU capability: `[12, 0]`
- Torch arch list: `['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']`
- Triton: `3.6.0`

## Tooling

- nvcc: `nvcc: NVIDIA (R) Cuda compiler driver Copyright (c) 2005-2023 NVIDIA Corporation Built on Fri_Jan__6_16:45:21_PST_2023 Cuda compilation tools, release 12.0, V12.0.140 Build cuda_12.0.r12.0/compiler.32267302_0`
- opencode: `1.14.22`
- KernelBench commit: `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`
- llama.cpp model endpoint: `{"models":[{"name":"gemma-4-12b-it-q4_k_m","model":"gemma-4-12b-it-q4_k_m","modified_at":"","size":"","digest":"","type":"model","description":"","tags":[""],"capabilities":["completion"],"parameters":"","details":{"parent_model":"","format":"gguf","family":"","families":[""],"parameter_size":"","quantization_level":""}}],"object":"list","data":[{"id":"gemma-4-12b-it-q4_k_m","aliases":["gemma-4-12b-it-q4_k_m"],"tags":[],"object":"model","created":1780599673,"owned_by":"llamacpp","meta":{"vocab_t`

## Native CUDA Extension Status

Native PyTorch CUDA extension evaluation is not the active backend for this run: the installed `nvcc` is CUDA 12.0 and fails on the RTX 5080 `compute_120` target. The local baseline therefore uses KernelBench-supported Triton kernels.

Raw environment JSON: `cuda_kernel_lab/reports/environment.json`
