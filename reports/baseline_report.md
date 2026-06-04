# Gemma 4 12B Q4 Kernel Baseline

Run: `gemma_q4_triton_reference_shapes_pinned`
Model: `gemma-4-12b-it-q4_k_m`
Benchmark scope: local reduced KernelBench-style Level 1/2 tasks using Triton backend.

Q8 comparison: attempted, but not feasible in this live run because the temporary Q8 server exited during load. See `cuda_kernel_lab/reports/q8_attempt.json`.

## Summary

- Tasks: 4
- Solved: 2
- Correctness rate (`fast_0`): 0.500
- Faster-than-PyTorch rate (`fast_1`): 0.250
- Attempt compile rate: 0.333
- Average speedup among correct attempts: 0.9534076570503166
- Best speedup: 1.009199648419306
- Failure types: `{'compile_error': 3, 'static_reject': 1}`

## Per Task

### affine_1d

- Name: 1D affine transform
- Backend: triton
- Solved: True
- Best attempt: 1
- Best speedup: 0.8976156656813272
- Attempts: 1

Attempt 1: compiled=True correct=True failure=None speedup=0.8976156656813272

### leaky_relu_1d

- Name: 1D LeakyReLU
- Backend: triton
- Solved: False
- Best attempt: None
- Best speedup: None
- Attempts: 2

Attempt 1: compiled=False correct=False failure=compile_error speedup=None

Attempt 2: compiled=False correct=False failure=compile_error speedup=None

### fused_square_relu_1d

- Name: 1D fused square plus ReLU
- Backend: triton
- Solved: True
- Best attempt: 1
- Best speedup: 1.009199648419306
- Attempts: 1

Attempt 1: compiled=True correct=True failure=None speedup=1.009199648419306

### row_mean_2d

- Name: 2D row mean reduction
- Backend: triton
- Solved: False
- Best attempt: None
- Best speedup: None
- Attempts: 2

Attempt 1: compiled=False correct=False failure=compile_error speedup=None

Attempt 2: compiled=False correct=False failure=static_reject speedup=None
