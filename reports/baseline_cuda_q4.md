# Gemma 4 12B Q4 Native CUDA Baseline

Combined native CUDA status across the reduced eight-task suite. The first four tasks used two bounded attempts; the four expanded tasks used one first-pass attempt.

- Tasks: 8
- Solved: 4
- Correctness (`fast_0`): 0.500
- Faster-than-PyTorch (`fast_1`): 0.250
- Compile rate: 0.556
- Attempts: 9
- Average speedup among correct: 1.05377228805201
- Best speedup: 1.672956043148107
- Failure types: `{'compile_error': 4, 'correctness_failure': 1}`

## Run `gemma_q4_cuda_cpp_baseline`

### affine_1d

- Solved: True
- Best speedup: 1.321888325170469
- Attempts: 1
- Attempt 1: compiled=True correct=True failure=None speedup=1.321888325170469

### leaky_relu_1d

- Solved: False
- Best speedup: None
- Attempts: 2
- Attempt 1: compiled=False correct=False failure=compile_error speedup=None
- Attempt 2: compiled=False correct=False failure=compile_error speedup=None

### fused_square_relu_1d

- Solved: True
- Best speedup: 1.672956043148107
- Attempts: 1
- Attempt 1: compiled=True correct=True failure=None speedup=1.672956043148107

### row_mean_2d

- Solved: True
- Best speedup: 0.6430636064240063
- Attempts: 1
- Attempt 1: compiled=True correct=True failure=None speedup=0.6430636064240063

## Run `gemma_q4_cuda_cpp_expanded_new_tasks`

### row_max_2d

- Solved: False
- Best speedup: None
- Attempts: 1
- Attempt 1: compiled=False correct=False failure=compile_error speedup=None

### row_softmax_2d

- Solved: False
- Best speedup: None
- Attempts: 1
- Attempt 1: compiled=True correct=False failure=correctness_failure speedup=None

### layer_norm_2d

- Solved: True
- Best speedup: 0.5771811774654574
- Attempts: 1
- Attempt 1: compiled=True correct=True failure=None speedup=0.5771811774654574

### matmul_2d

- Solved: False
- Best speedup: None
- Attempts: 1
- Attempt 1: compiled=False correct=False failure=compile_error speedup=None
