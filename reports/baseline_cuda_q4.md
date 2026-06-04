# Gemma 4 12B Q4 Native CUDA Baseline

Native CUDA here means CUDA C++ compiled through `torch.utils.cpp_extension` with the repo-local CUDA 13.0 `nvcc`, targeting RTX 5080 `sm_120`. The evaluator rejects `cuda_cpp` candidates that omit `load_inline`, `CUDA_SOURCE`, `CPP_SOURCE`, `PYBIND11_MODULE`, or `functions=None`.

## Original 8-Task Eval

Clean first pass:

- Runs: `gemma_q4_cuda_cpp_baseline`, `gemma_q4_cuda_cpp_expanded_new_tasks`
- Tasks: 8
- Solved: 4
- Correctness (`fast_0`): 0.500
- Faster-than-PyTorch (`fast_1`): 0.250
- Compile rate: 0.556
- Attempts: 9
- Average speedup among correct: 1.05377228805201
- Best speedup: 1.672956043148107
- Failure types: `{'compile_error': 4, 'correctness_failure': 1}`

Best-known after targeted retries:

- Additional run: `gemma_q4_cuda_cpp_retry_failed_tasks`
- Solved: 6/8
- Correctness (`best-known fast_0`): 0.750
- Faster-than-PyTorch (`best-known fast_1`): 0.250
- Still unresolved: `leaky_relu_1d`, `matmul_2d`

## Train-Trace Runs

| Run | Tasks | Solved | `fast_1` | Attempts | Compile rate | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `gemma_q4_cuda_cpp_train_trace_v1` | 8 | 5 | 0.500 | 14 | 0.357 | useful success batch |
| `gemma_q4_cuda_cpp_train_trace_v2` | 8 | 2 | 0.000 | 15 | 0.133 | mostly indentation compile errors |
| `gemma_q4_cuda_cpp_train_trace_v2_repair` | 6 | 0 | 0.000 | 12 | 0.000 | targeted repair hints did not help |
| `gemma_q4_cuda_cpp_mechanical_repair_v1` | 24 | 13 | 0.167 | 40 | 0.875 | deterministic repairs, not raw pass@N |
| `gemma_q4_cuda_cpp_train_trace_v3` | 8 | 0 | 0.000 | 16 | 0.000 | raw generations all failed compile |
| `gemma_q4_cuda_cpp_train_trace_v3_mechanical_repair` | 8 | 3 | 0.250 | 16 | 0.438 | deterministic repairs, not raw pass@N |
| `gemma_q4_cuda_cpp_unsolved_normalized_v1` | 8 | 3 | 0.250 | 14 | 0.500 | extraction normalization, not strict raw pass@N |

Train-trace v2 successes:

| Task | Solved | Best speedup |
| --- | ---: | ---: |
| `abs_1d` | yes | 0.884861400766101 |
| `binary_sub_1d` | yes | 0.8430107167500398 |

## Current Native CUDA Coverage

Best-known across the 32 registered tasks:

- Raw solved: 13/32
- Extraction-normalized solved: 3/32
- Mechanical repair solved: 16/32
- Best-known with provenance-labeled repaired/normalized candidates: 27/32
- Raw faster-than-PyTorch rows: 6
- Average speedup among correct: 0.9074927331774545
- Best speedup: 1.672956043148107
- Raw native CUDA clean success traces: 13
- Mechanically repaired success traces: 30
- Extraction-normalized success traces: 3
- Curated candidate rows with repair/normalization provenance: 46
- Still unsolved after raw plus repair/normalization: `column_mul_add_2d`, `column_relu_bias_2d`, `matmul_2d`, `tanh_1d`, `weighted_sum_1d`

This is real progress, but raw Gemma success data is still below the 50-200 clean native CUDA traces needed before QLoRA is justified. Repaired rows are useful only with provenance labels.
