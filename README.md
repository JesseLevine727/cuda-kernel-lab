# CUDA Kernel Lab

This directory contains a local KernelBench-style workflow for testing Gemma 4 12B on GPU kernel generation.

The current local scope targets native CUDA C++ generation first. Triton remains available as a comparison backend only.

## Current State

- GPU: RTX 5080 16GB, `sm_120`
- Native CUDA toolkit: repo-local CUDA 13.0 in `cuda_kernel_lab/toolchains/cuda-13.0-local/`
- CUDA env: `source cuda_kernel_lab/env/cuda.sh`
- Model: pinned Gemma 4 12B Q4 GGUF through llama.cpp
- Server: `http://127.0.0.1:8080/v1`
- Q4 alias: `gemma-4-12b-it-q4_k_m`
- Q8: pinned Q8 loads at 4k context and passes trivial generation sanity; long-context tests remain open.

Pinned Q4:

```text
/home/elfo/.cache/huggingface/hub/models--ggml-org--gemma-4-12B-it-GGUF/snapshots/0f3915622134b2b6279d02f482cb12adc3d9ca3d/gemma-4-12B-it-Q4_K_M.gguf
```

Do not switch back to a moving Hugging Face GGUF ref without testing generation first. On this machine, a newer cached Q4 snapshot emitted repeated `<unused49>` tokens.

## Commands

Run native CUDA trace collection:

```bash
python3 -m cuda_kernel_lab.run_gemma_eval \
  --backend cuda_cpp \
  --tasks vector_add_1d,vector_mul_1d,axpy_1d,sigmoid_1d,clamp_1d,row_sum_2d,row_min_2d,column_scale_2d \
  --run-name gemma_q4_cuda_cpp_train_trace_v1 \
  --max-attempts 2
```

Create dataset records from a run:

```bash
python3 -m cuda_kernel_lab.make_dataset cuda_kernel_lab/runs/<run-name>/results.json
python3 -m cuda_kernel_lab.make_dataset cuda_kernel_lab/runs/<run-name>/results.json --successful-only
```

Evaluate one candidate:

```bash
python3 -m cuda_kernel_lab.kernel_eval \
  --task-id affine_1d \
  --source cuda_kernel_lab/examples/affine_cuda_cpp.py \
  --backend cuda_cpp
```

## Scope

- The official KernelBench repo is cloned at `external/KernelBench`.
- This harness records prompts, raw responses, extracted candidate code, evaluator output, correctness, speed, and failure types.
- `cuda_cpp` candidates must contain `load_inline`, `CUDA_SOURCE`, `CPP_SOURCE`, `PYBIND11_MODULE`, and `functions=None`.
- Current native CUDA clean success traces: 13.
- QLoRA should wait until there are 50-200 clean native CUDA success traces with held-out eval tasks excluded.
