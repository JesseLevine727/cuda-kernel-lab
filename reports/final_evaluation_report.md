# Final Evaluation Report

Status: significant progress, but active goal remains open.

## Native CUDA Q4 Combined

- Tasks: 8
- Solved: 4
- Correctness (`fast_0`): 0.500
- Faster-than-PyTorch (`fast_1`): 0.250
- Compile rate: 0.556
- Attempts: 9
- Average speedup among correct: 1.05377228805201
- Best speedup: 1.672956043148107
- Failure types: `{'compile_error': 4, 'correctness_failure': 1}`

## Triton Q4 Current Comparison

- Tasks: 4
- Solved: 2
- Correctness (`fast_0`): 0.500
- Faster-than-PyTorch (`fast_1`): 0.250
- Compile rate: 0.714
- Attempts: 7
- Average speedup among correct: 0.8389367334885676
- Best speedup: 1.0761904614830013
- Failure types: `{'runtime_error': 3, 'compile_error': 2}`

## Decision

Native CUDA support is real now: repo-local CUDA 13.0 nvcc compiles `sm_120`, PyTorch extensions compile with `compute_120/sm_120`, and Gemma-generated CUDA candidates are evaluated. The current dataset is too small for fine-tuning. Next action is trace collection and prompt/retry hardening.