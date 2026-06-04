# Mechanical Repair v1

This run applies deterministic fixes to failed native CUDA candidates, then re-evaluates them with the same `cuda_cpp` evaluator. It is useful for dataset curation and failure analysis, but it is not raw Gemma pass@N.

## Run

- Run: `cuda_kernel_lab/runs/gemma_q4_cuda_cpp_mechanical_repair_v1`
- Inputs: baseline, expanded tasks, retry, train-trace v1, train-trace v2, and prompt-repair results.
- Attempts repaired/evaluated: 40
- Unique tasks solved: 13/24
- Correct repaired rows: 26
- Faster-than-PyTorch task rate: 4/24
- Compile rate: 0.875
- Average best speedup among solved tasks: 0.9706559884698732
- Best speedup: 1.6010781125624762

## Coverage

Repair solved these tasks:

`axpy_1d`, `binary_sub_1d`, `clamp_1d`, `column_bias_2d`, `negate_1d`, `relu_bias_1d`, `row_max_2d`, `row_mean_256_2d`, `row_min_2d`, `scalar_add_1d`, `sigmoid_1d`, `square_1d`, `vector_mul_1d`.

New task coverage beyond raw Gemma successes:

`axpy_1d`, `column_bias_2d`, `negate_1d`, `relu_bias_1d`, `row_mean_256_2d`, `row_min_2d`, `scalar_add_1d`, `square_1d`.

Still unsolved in the 24-task suite after raw plus repair:

`column_scale_2d`, `leaky_relu_1d`, `matmul_2d`.

## Datasets

- `cuda_kernel_lab/datasets/gemma_q4_cuda_cpp_mechanical_repair_v1_all.jsonl`: 40 rows
- `cuda_kernel_lab/datasets/gemma_q4_cuda_cpp_mechanical_repair_v1_success.jsonl`: 26 rows

Labels:

- `mechanical_repair_correct_but_slower`: 19
- `mechanical_repair_correct_and_faster`: 7
- `mechanical_repair_correctness_fix_needed`: 9
- `mechanical_repair_compile_fix_needed`: 5

Use the repair rows with provenance labels. Do not merge them silently with raw Gemma success rows.
