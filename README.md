# CUDA Kernel Lab

This directory contains a local KernelBench-style workflow for testing Gemma 4 12B on GPU kernel generation.

The current local scope is a reduced Level 1/2 task set using Triton, because this machine has an RTX 5080 with PyTorch CUDA 13.0 support but the installed `nvcc` is CUDA 12.0 and cannot compile `compute_120` native CUDA extensions.

## Commands

Probe the environment:

```bash
python3 -m cuda_kernel_lab.probe_env
```

Run a Gemma Q4 baseline:

```bash
python3 -m cuda_kernel_lab.run_gemma_eval --max-attempts 2
```

Create dataset records from a run:

```bash
python3 -m cuda_kernel_lab.make_dataset cuda_kernel_lab/runs/<run-name>/results.json
```

Write reports:

```bash
python3 -m cuda_kernel_lab.write_reports \
  cuda_kernel_lab/runs/<run-name>/results.json \
  --dataset cuda_kernel_lab/datasets/<run-name>.jsonl
```

## Scope

- The official KernelBench repo is cloned at `external/KernelBench`.
- This harness records prompts, raw responses, extracted candidate code, evaluator output, correctness, speed, and failure types.
- It is intended to decide whether QLoRA is worth doing after prompt/retry loops are measured.

