# Fine-Tuning Feasibility

Decision: **Do not fine-tune yet. The immediate blocker is evaluation/tooling maturity, not enough curated data.**

## Evidence

- GPU: NVIDIA GeForce RTX 5080 capability [12, 0]
- PyTorch: 2.11.0+cu130 CUDA 13.0
- Triton: 3.6.0
- nvcc: nvcc: NVIDIA (R) Cuda compiler driver Copyright (c) 2005-2023 NVIDIA Corporation Built on Fri_Jan__6_16:45:21_PST_2023 Cuda compilation tools, release 12.0, V12.0.140 Build cuda_12.0.r12.0/compiler.32267302_0
- Baseline solved 2/4 tasks.
- Dataset records created from eval loop: 6
- Correct dataset records: 2
- Dataset labels: `{'correct_but_slower': 1, 'compile_fix_needed': 3, 'correct_and_faster': 1, 'static_reject': 1}`
- Q8 comparison was attempted but not feasible in the live run; the temporary Q8 server exited during load and Q4 was restored.

## Implications

- The original HF safetensors checkpoint is required for LoRA/QLoRA training; the current GGUF is inference-only.
- The local CUDA extension path is not ready because the installed nvcc cannot compile `compute_120` for the RTX 5080.
- Triton gives a working local GPU-kernel backend and should be used to collect traces first.
- A useful first fine-tune should wait until there are at least 50-200 clean CUDA/Triton examples with held-out eval tasks excluded.

## Next Gate

Collect more successful traces or install a CUDA 13 toolkit for native CUDA extension evaluation, then rerun the baseline before QLoRA.
