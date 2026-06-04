# Retry Loop Report

## Native CUDA Original First Pass

- Runs: `gemma_q4_cuda_cpp_baseline`, `gemma_q4_cuda_cpp_expanded_new_tasks`
- Tasks: 8
- Solved: 4
- Correctness (`fast_0`): 0.500
- Faster-than-PyTorch (`fast_1`): 0.250
- Compile rate: 0.556
- Attempts: 9
- Failure types: `{'compile_error': 4, 'correctness_failure': 1}`

## Targeted Native CUDA Retries

- Run: `gemma_q4_cuda_cpp_retry_failed_tasks`
- Tasks retried: `leaky_relu_1d`, `row_max_2d`, `row_softmax_2d`, `matmul_2d`
- Solved: 2/4
- Attempts: 10
- Compile rate: 0.300
- Successes: `row_max_2d`, `row_softmax_2d`
- Still failing: `leaky_relu_1d`, `matmul_2d`

## Train-Trace v1

- Run: `gemma_q4_cuda_cpp_train_trace_v1`
- Tasks: 8
- Solved: 5
- Attempts: 14
- Compile rate: 0.357
- Successes: `vector_add_1d`, `vector_mul_1d`, `sigmoid_1d`, `clamp_1d`, `row_sum_2d`

## Train-Trace v2

- Run: `gemma_q4_cuda_cpp_train_trace_v2`
- Tasks: 8
- Solved: 2
- Attempts: 15
- Compile rate: 0.133
- Successes: `abs_1d`, `binary_sub_1d`
- Dominant failure: `IndentationError: unexpected indent`

## Train-Trace v2 Focused Repair

- Run: `gemma_q4_cuda_cpp_train_trace_v2_repair`
- Tasks: 6
- Solved: 0
- Attempts: 12
- Compile rate: 0.000

The focused repair hints did not overcome the repeated top-level indentation and malformed-module pattern.

## Mechanical Repair

- Run: `gemma_q4_cuda_cpp_mechanical_repair_v1`
- Inputs: all native CUDA failed attempts from baseline, expanded tasks, retry, train-trace v1, train-trace v2, and prompt repair.
- Attempts: 40
- Unique tasks solved: 13/24
- Correct repaired rows: 26
- New task coverage beyond raw Gemma successes: `axpy_1d`, `column_bias_2d`, `negate_1d`, `relu_bias_1d`, `row_mean_256_2d`, `row_min_2d`, `scalar_add_1d`, `square_1d`

## Train-Trace v3

- Run: `gemma_q4_cuda_cpp_train_trace_v3`
- Tasks: 8
- Solved: 0
- Attempts: 16
- Compile rate: 0.000
- Dominant failure: repeated malformed native CUDA module syntax.

## Train-Trace v3 Mechanical Repair

- Run: `gemma_q4_cuda_cpp_train_trace_v3_mechanical_repair`
- Tasks: 8
- Solved: 3
- Correct repaired rows: 4
- Successes: `scalar_mul_1d`, `addcmul_1d`, `threshold_1d`

## Extraction-Normalized Unsolved Retry

- Run: `gemma_q4_cuda_cpp_unsolved_normalized_v1`
- Tasks: 8
- Solved: 3
- Correct normalized rows: 3
- Successes: `column_scale_2d`, `leaky_relu_1d`, `relu_mul_1d`
- Still unresolved: `column_mul_add_2d`, `column_relu_bias_2d`, `matmul_2d`, `tanh_1d`, `weighted_sum_1d`

## Triton Comparison

- Runs: `gemma_q4_triton_comparison_current`, `gemma_q4_triton_expanded_new_tasks`
- Tasks: 8
- Solved: 2
- Correctness (`fast_0`): 0.250
- Faster-than-PyTorch (`fast_1`): 0.125
- Attempts: 11
- Failure types: `{'runtime_error': 3, 'compile_error': 5, 'static_reject': 1}`

## Decision

Retry feedback is useful but weak for Gemma Q4 native CUDA. Mechanical repair recovers useful candidates, but the next model-side improvement should be a stronger generation scaffold and more raw native CUDA trace collection.
