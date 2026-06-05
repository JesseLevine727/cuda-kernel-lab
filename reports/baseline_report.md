# Gemma 4 12B Q4 Kernel Baseline

This file is a compact index of the current baselines. Detailed numbers live in:

- `cuda_kernel_lab/reports/baseline_cuda_q4.md`
- `cuda_kernel_lab/reports/baseline_triton_q4.md`
- `cuda_kernel_lab/reports/train_trace_v1.md`
- `cuda_kernel_lab/reports/train_trace_v2.md`
- `cuda_kernel_lab/reports/q8_feasibility.md`

## Native CUDA C++ / nvcc

- Model: `gemma-4-12b-it-q4_k_m`
- Backend: `cuda_cpp`
- Original 8-task clean first pass: 4/8 solved, `fast_1` 0.250
- Original 8-task best-known after retries: 6/8 solved, `fast_1` 0.250
- Train-trace v1: 5/8 solved, `fast_1` 0.500
- Train-trace v2: 2/8 solved, `fast_1` 0.000
- Train-trace v3: 0/8 solved raw, `fast_1` 0.000
- Train-trace v2 prompt repair: 0/6 solved
- Mechanical repair total: 16/32 solved, 30 correct repaired rows
- Extraction-normalized retry: 3/8 solved, 3 correct normalized rows
- Raw best-known across 32 registered tasks: 13/32 solved
- Best-known with provenance-labeled repaired/normalized candidates: 27/32 solved
- Raw native CUDA clean success traces: 13
- Curated candidate rows with repair/normalization provenance: 46
- Best speedup: 1.672956043148107

## Triton Comparison

- Model: `gemma-4-12b-it-q4_k_m`
- Backend: `triton`
- Original comparison tasks: 8
- Solved: 2
- `fast_1`: 0.125
- Best speedup: 1.0761904614830013

## Q8

Pinned Q8 is feasible at 4k context and passed trivial generation sanity. It has not yet been benchmarked for kernel-generation quality or long-context stability.
