# Gemma 4 12B Q4 Triton Comparison Baseline

This run keeps Triton as a comparison backend. It is not counted as proof of native CUDA support.

- Run: `gemma_q4_triton_comparison_current`
- Backend: `triton`
- Tasks: 4
- Correctness (`fast_0`): 0.500
- Faster-than-PyTorch (`fast_1`): 0.250
- Compile rate: 0.714
- Attempts: 7
- Average speedup among correct: 0.8389367334885676
- Best speedup: 1.0761904614830013
- Failure types: `{'runtime_error': 3, 'compile_error': 2}`

## Per Task

### affine_1d

- Solved: False
- Best speedup: None
- Attempts: 2

Attempt 1: compiled=True correct=False failure=runtime_error speedup=None
Attempt 2: compiled=True correct=False failure=runtime_error speedup=None

### leaky_relu_1d

- Solved: False
- Best speedup: None
- Attempts: 2

Attempt 1: compiled=False correct=False failure=compile_error speedup=None
Attempt 2: compiled=True correct=False failure=runtime_error speedup=None

### fused_square_relu_1d

- Solved: True
- Best speedup: 1.0761904614830013
- Attempts: 1

Attempt 1: compiled=True correct=True failure=None speedup=1.0761904614830013

### row_mean_2d

- Solved: True
- Best speedup: 0.6016830054941337
- Attempts: 2

Attempt 1: compiled=False correct=False failure=compile_error speedup=None
Attempt 2: compiled=True correct=True failure=None speedup=0.6016830054941337
