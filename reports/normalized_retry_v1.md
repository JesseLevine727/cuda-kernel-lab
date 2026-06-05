# Extraction-Normalized Retry v1

Run: `gemma_q4_cuda_cpp_unsolved_normalized_v1`

Purpose: retry the eight still-unsolved tasks with opt-in extraction normalization. This normalization removes common code-block extraction damage and records provenance in each attempt; it is not strict raw pass@N and is separate from post-hoc mechanical repair.

## Result

- Backend: `cuda_cpp`
- Tasks: 8
- Solved: 3
- Correct extraction-normalized rows: 3
- Faster-than-PyTorch task rate: 0.250
- Attempts: 14
- Compile rate: 0.500
- Failure types: `{'compile_error': 7, 'correctness_failure': 4}`
- Average speedup among solved tasks: 1.0779603951886971
- Best speedup: 1.2909090548299624

Solved:

`column_scale_2d`, `leaky_relu_1d`, `relu_mul_1d`.

Still unresolved after raw plus repair plus extraction-normalized retry:

`column_mul_add_2d`, `column_relu_bias_2d`, `matmul_2d`, `tanh_1d`, `weighted_sum_1d`.

## Datasets

- `cuda_kernel_lab/datasets/gemma_q4_cuda_cpp_unsolved_normalized_v1_all.jsonl`: 14 rows
- `cuda_kernel_lab/datasets/gemma_q4_cuda_cpp_unsolved_normalized_v1_success.jsonl`: 3 rows

Labels:

- `extraction_normalized_correct_and_faster`: 2
- `extraction_normalized_correct_but_slower`: 1
- `extraction_normalized_compile_fix_needed`: 7
- `extraction_normalized_correctness_fix_needed`: 4

Use these rows with `normalization` provenance intact.
