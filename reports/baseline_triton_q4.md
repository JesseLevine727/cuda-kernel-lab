# Gemma 4 12B Q4 Triton Comparison Baseline

Triton is kept as a comparison backend only. It is not counted as proof of native CUDA C++/`nvcc` support.

Runs:

- `gemma_q4_triton_comparison_current`
- `gemma_q4_triton_expanded_new_tasks`

## Eight-Task Summary

- Backend: `triton`
- Tasks: 8
- Solved: 2
- Correctness (`fast_0`): 0.250
- Faster-than-PyTorch (`fast_1`): 0.125
- Compile rate: 0.455
- Attempts: 11
- Average speedup among correct: 0.8389367334885676
- Best speedup: 1.0761904614830013
- Failure types: `{'runtime_error': 3, 'compile_error': 5, 'static_reject': 1}`

## Per-Task Result

| Task | Solved | Best speedup | Run |
| --- | ---: | ---: | --- |
| `affine_1d` | no | n/a | `gemma_q4_triton_comparison_current` |
| `leaky_relu_1d` | no | n/a | `gemma_q4_triton_comparison_current` |
| `fused_square_relu_1d` | yes | 1.0761904614830013 | `gemma_q4_triton_comparison_current` |
| `row_mean_2d` | yes | 0.6016830054941337 | `gemma_q4_triton_comparison_current` |
| `row_max_2d` | no | n/a | `gemma_q4_triton_expanded_new_tasks` |
| `row_softmax_2d` | no | n/a | `gemma_q4_triton_expanded_new_tasks` |
| `layer_norm_2d` | no | n/a | `gemma_q4_triton_expanded_new_tasks` |
| `matmul_2d` | no | n/a | `gemma_q4_triton_expanded_new_tasks` |

## Interpretation

On this reduced suite, native CUDA currently has better correctness coverage than Triton after retries: 6/8 best-known native CUDA versus 2/8 Triton. Triton still provides useful comparison traces and failure examples, but it should not be the training target if the goal is CUDA C++ kernel generation.
