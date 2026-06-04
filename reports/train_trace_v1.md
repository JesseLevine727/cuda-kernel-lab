# Native CUDA Train-Trace v1

Run: `gemma_q4_cuda_cpp_train_trace_v1`

Purpose: expand trace-collection data without contaminating the original held-out eval tasks.

## Summary

- Backend: `cuda_cpp`
- Tasks: 8
- Solved: 5
- Correctness (`fast_0`): 0.625
- Faster-than-PyTorch (`fast_1`): 0.500
- Attempts: 14
- Compile rate: 0.357
- Failure types: `{'compile_error': 9}`
- Average speedup among correct: 1.0001169718468643
- Best speedup: 1.0271738820965892

## Per Task

| Task | Solved | Attempts | Best speedup | Notes |
| --- | ---: | ---: | ---: | --- |
| `vector_add_1d` | yes | 1 | 1.0271738820965892 | correct and faster |
| `vector_mul_1d` | yes | 2 | 1.002570676296718 | attempt 1 compile error, attempt 2 fixed |
| `axpy_1d` | no | 2 | n/a | compile errors |
| `sigmoid_1d` | yes | 2 | 1.0240641591590394 | attempt 1 compile error, attempt 2 fixed |
| `clamp_1d` | yes | 2 | 0.9388185600898313 | correct but slower |
| `row_sum_2d` | yes | 1 | 1.007957581592144 | correct and faster |
| `row_min_2d` | no | 2 | n/a | compile errors |
| `column_scale_2d` | no | 2 | n/a | compile errors |

## Dataset Outputs

- `cuda_kernel_lab/datasets/gemma_q4_cuda_cpp_train_trace_v1_all.jsonl`: 14 records
- `cuda_kernel_lab/datasets/gemma_q4_cuda_cpp_train_trace_v1_success.jsonl`: 5 records
