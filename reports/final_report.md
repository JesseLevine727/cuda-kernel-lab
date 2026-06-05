# Final Kernel Workflow Report

This is the current local gate report for the Gemma 4 12B CUDA-kernel project.

## What Was Built

- Local reduced KernelBench-style harness under `cuda_kernel_lab/`.
- Backend-aware Gemma generation loop with retry feedback.
- Native CUDA C++ evaluator using repo-local CUDA 13.0 `nvcc`.
- Strict `cuda_cpp` static gate requiring native CUDA extension structure.
- Triton evaluator retained as a comparison backend.
- Subprocess evaluator for compile/import, multi-seed correctness, timing, speedup, and failure labels.
- JSONL trace dataset builder.
- Environment, CUDA toolchain, baseline, Q8, dataset, retry, and fine-tuning feasibility reports.

## Current Result

- Model: `gemma-4-12b-it-q4_k_m`
- Native CUDA original clean first pass: 4/8 solved, `fast_1` 0.250.
- Native CUDA original best-known after retries: 6/8 solved, `fast_1` 0.250.
- Native CUDA train-trace v1: 5/8 solved, `fast_1` 0.500.
- Native CUDA train-trace v2: 2/8 solved, `fast_1` 0.000.
- Native CUDA train-trace v3: 0/8 solved raw, `fast_1` 0.000.
- Native CUDA v2 focused prompt repair: 0/6 solved.
- Native CUDA mechanical repair total: 16/32 solved, 30 correct repaired rows.
- Native CUDA extraction-normalized retry: 3/8 solved, 3 correct normalized rows.
- Native CUDA raw best-known across registered tasks: 13/32 solved.
- Native CUDA best-known with provenance-labeled repaired/normalized candidates: 27/32 solved.
- Triton comparison: 2/8 solved, `fast_1` 0.125.
- Raw native CUDA clean success traces available: 13.
- Curated candidate rows with repair/normalization provenance: 46.
- Q8 comparison: pinned Q8 loads at 4k context and passes trivial generation sanity; long-context and quality tests remain open.

## Fine-Tuning Decision

Do not QLoRA fine-tune yet. The native CUDA workflow is real, but the raw dataset is still below the 50-200 clean native CUDA success traces needed before training is worth attempting. Even the provenance-labeled curated set is still below 50.

The next useful work is more raw trace collection, better prompt-side repair, and careful use of provenance-labeled mechanical repair data.
