# Gemma 4 12B Q4 Kernel Baseline

Run: `gemma_q4_triton_baseline`
Model: `gemma-4-12b-it-q4_k_m`
Benchmark scope: local reduced KernelBench-style Level 1/2 tasks using Triton backend.

Q8 comparison: attempted, but not feasible in this live run because the temporary Q8 server exited during load. See `cuda_kernel_lab/reports/q8_attempt.json`.

## Summary

- Tasks: 4
- Solved: 3
- Correctness rate (`fast_0`): 0.750
- Faster-than-PyTorch rate (`fast_1`): 0.250
- Attempt compile rate: 0.833
- Average speedup among correct attempts: 0.7581592916540512
- Best speedup: 1.0580912854719033
- Failure types: `{'runtime_error': 2, 'compile_error': 1}`

## Per Task

### affine_1d

- Name: 1D affine transform
- Backend: triton
- Solved: False
- Best attempt: None
- Best speedup: None
- Attempts: 2

Attempt 1: compiled=True correct=False failure=runtime_error speedup=None

Attempt 2: compiled=True correct=False failure=runtime_error speedup=None

### leaky_relu_1d

- Name: 1D LeakyReLU
- Backend: triton
- Solved: True
- Best attempt: 2
- Best speedup: 0.6568627809976137
- Attempts: 2

Attempt 1: compiled=False correct=False failure=compile_error speedup=None

Attempt 2: compiled=True correct=True failure=None speedup=0.6568627809976137

### fused_square_relu_1d

- Name: 1D fused square plus ReLU
- Backend: triton
- Solved: True
- Best attempt: 1
- Best speedup: 1.0580912854719033
- Attempts: 1

Attempt 1: compiled=True correct=True failure=None speedup=1.0580912854719033

### row_mean_2d

- Name: 2D row mean reduction
- Backend: triton
- Solved: True
- Best attempt: 1
- Best speedup: 0.5595238084926364
- Attempts: 1

Attempt 1: compiled=True correct=True failure=None speedup=0.5595238084926364
