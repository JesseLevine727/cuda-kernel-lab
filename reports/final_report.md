# Final Kernel Workflow Report

This is the completed first local gate for the Gemma 4 12B CUDA-kernel fine-tuning project.

## What Was Built

- Official KernelBench cloned under `external/KernelBench`.
- Local reduced KernelBench-style Triton harness under `cuda_kernel_lab/`.
- Bounded Gemma generation loop with retry feedback.
- Subprocess evaluator for compile/import, multi-seed correctness, timing, speedup, and failure labels.
- JSONL trace dataset builder.
- Environment, baseline, and fine-tuning feasibility reports.

## Baseline Result

- Model: `gemma-4-12b-it-q4_k_m`
- Tasks: 4
- Correctness rate (`fast_0`): 0.500
- Faster-than-PyTorch rate (`fast_1`): 0.250
- Correct traces available: 2
- Q8 comparison: attempted but not feasible in this live run; Q4 was restored.

## Fine-Tuning Decision

Do not QLoRA fine-tune yet. The current run proves the local eval loop works, but it only produced a tiny number of correct examples and the native CUDA-extension backend is blocked by the CUDA 12.0 `nvcc` toolchain on compute capability 12.0.

The next useful work is trace collection and/or a CUDA 13 toolkit install, not immediate training.

## Before/After Status

No tuned adapter was produced because the fine-tuning gate failed. The before/after comparison is therefore intentionally deferred; the baseline artifacts here are the reference point for any future tuned model.
