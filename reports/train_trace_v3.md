# Native CUDA Train-Trace v3

Run: `gemma_q4_cuda_cpp_train_trace_v3`

Purpose: expand native CUDA trace collection with eight additional simple elementwise and column-broadcast tasks.

## Raw Q4 Run

- Backend: `cuda_cpp`
- Tasks: 8
- Solved: 0
- Correctness (`fast_0`): 0.000
- Faster-than-PyTorch (`fast_1`): 0.000
- Attempts: 16
- Compile rate: 0.000
- Failure types: `{'compile_error': 16}`

The raw generations repeated the malformed native CUDA module pattern: top-level indentation, extra closing parentheses in `return ext...`, and occasional invalid CUDA/C++ lines.

## Mechanical Repair Follow-Up

Run: `gemma_q4_cuda_cpp_train_trace_v3_mechanical_repair`

- Attempts: 16
- Solved: 3/8
- Correct repaired rows: 4
- Faster-than-PyTorch task rate: 0.250
- Compile rate: 0.4375
- Average best speedup among solved tasks: 1.6055426134367439
- Best speedup: 2.024064199004213

Solved with repair:

`scalar_mul_1d`, `addcmul_1d`, `threshold_1d`.

Still unresolved in v3:

`weighted_sum_1d`, `relu_mul_1d`, `tanh_1d`, `column_mul_add_2d`, `column_relu_bias_2d`.

`relu_mul_1d` was later solved by `gemma_q4_cuda_cpp_unsolved_normalized_v1` using extraction normalization.

## Dataset Outputs

- `cuda_kernel_lab/datasets/gemma_q4_cuda_cpp_train_trace_v3_all.jsonl`: 16 records
- `cuda_kernel_lab/datasets/gemma_q4_cuda_cpp_train_trace_v3_success.jsonl`: 0 records
- `cuda_kernel_lab/datasets/gemma_q4_cuda_cpp_train_trace_v3_mechanical_repair_all.jsonl`: 16 records
- `cuda_kernel_lab/datasets/gemma_q4_cuda_cpp_train_trace_v3_mechanical_repair_success.jsonl`: 4 records

Use the repaired rows only with provenance labels. They are not raw Gemma successes.
