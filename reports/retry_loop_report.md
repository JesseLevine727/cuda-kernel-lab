# Retry Loop Report

## Native CUDA Combined

- Tasks: 8
- Solved: 4
- Correctness (`fast_0`): 0.500
- Faster-than-PyTorch (`fast_1`): 0.250
- Compile rate: 0.556
- Attempts: 9
- Average speedup among correct: 1.05377228805201
- Best speedup: 1.672956043148107
- Failure types: `{'compile_error': 4, 'correctness_failure': 1}`

Native CUDA currently has real compile/runtime feedback and solved 4/8 tasks. Failures include compile errors and one correctness failure. More prompt work and retries are needed.

## Triton Comparison

- Tasks: 4
- Solved: 2
- Correctness (`fast_0`): 0.500
- Faster-than-PyTorch (`fast_1`): 0.250
- Compile rate: 0.714
- Attempts: 7
- Average speedup among correct: 0.8389367334885676
- Best speedup: 1.0761904614830013
- Failure types: `{'runtime_error': 3, 'compile_error': 2}`

Triton solved 2/4 in the current comparison run.

## Decision

Retry feedback is useful but not sufficient yet; collect more native CUDA fix traces before QLoRA.