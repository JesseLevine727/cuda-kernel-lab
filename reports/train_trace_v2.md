# Native CUDA Train-Trace v2

Run: `gemma_q4_cuda_cpp_train_trace_v2`

Purpose: expand trace-collection data with additional simple elementwise and broadcast tasks.

## Summary

- Backend: `cuda_cpp`
- Tasks: 8
- Solved: 2
- Correctness (`fast_0`): 0.250
- Faster-than-PyTorch (`fast_1`): 0.000
- Attempts: 15
- Compile rate: 0.133
- Failure types: `{'compile_error': 13}`
- Average speedup among correct: 0.8639360587580704
- Best speedup: 0.884861400766101

## Per Task

| Task | Solved | Attempts | Best speedup | Notes |
| --- | ---: | ---: | ---: | --- |
| `scalar_add_1d` | no | 2 | n/a | compile errors |
| `negate_1d` | no | 2 | n/a | compile errors |
| `square_1d` | no | 2 | n/a | compile errors |
| `abs_1d` | yes | 1 | 0.884861400766101 | correct but slower |
| `relu_bias_1d` | no | 2 | n/a | compile errors |
| `binary_sub_1d` | yes | 2 | 0.8430107167500398 | attempt 1 compile error, attempt 2 fixed |
| `column_bias_2d` | no | 2 | n/a | compile errors |
| `row_mean_256_2d` | no | 2 | n/a | compile errors |

## Focused Repair

Run: `gemma_q4_cuda_cpp_train_trace_v2_repair`

- Tasks: 6
- Solved: 0
- Attempts: 12
- Compile rate: 0.000
- Failure types: `{'compile_error': 12}`

The added targeted prompt hints did not fix the repeated indentation/malformed-module pattern.

## Mechanical Repair Follow-Up

Run: `gemma_q4_cuda_cpp_mechanical_repair_v1`

- Inputs included the v2 and v2 focused-repair failures.
- Correct repaired rows from v2/v2-repair families include `scalar_add_1d`, `negate_1d`, `square_1d`, `relu_bias_1d`, `column_bias_2d`, and `row_mean_256_2d`.
- These are deterministic post-processing repairs, not raw Gemma pass@N.

## Dataset Outputs

- `cuda_kernel_lab/datasets/gemma_q4_cuda_cpp_train_trace_v2_all.jsonl`: 15 records
- `cuda_kernel_lab/datasets/gemma_q4_cuda_cpp_train_trace_v2_success.jsonl`: 2 records
- `cuda_kernel_lab/datasets/gemma_q4_cuda_cpp_train_trace_v2_repair_all.jsonl`: 12 records
- `cuda_kernel_lab/datasets/gemma_q4_cuda_cpp_train_trace_v2_repair_success.jsonl`: 0 records
- `cuda_kernel_lab/datasets/gemma_q4_cuda_cpp_mechanical_repair_v1_all.jsonl`: 40 records
- `cuda_kernel_lab/datasets/gemma_q4_cuda_cpp_mechanical_repair_v1_success.jsonl`: 26 records
