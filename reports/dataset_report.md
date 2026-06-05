# Dataset Report

Current datasets are local trace datasets with prompts, raw generations, extracted code, evaluator results, labels, and provenance. They prove the workflow, but they remain too small for QLoRA.

## Native CUDA Datasets

| Dataset | Records | Correct | Labels |
| --- | ---: | ---: | --- |
| `gemma_q4_cuda_cpp_baseline_all.jsonl` | 5 | 3 | `correct_and_faster`: 2, `compile_fix_needed`: 2, `correct_but_slower`: 1 |
| `gemma_q4_cuda_cpp_baseline_success.jsonl` | 3 | 3 | `correct_and_faster`: 2, `correct_but_slower`: 1 |
| `gemma_q4_cuda_cpp_expanded_new_tasks_all.jsonl` | 4 | 1 | `compile_fix_needed`: 2, `correctness_fix_needed`: 1, `correct_but_slower`: 1 |
| `gemma_q4_cuda_cpp_expanded_new_tasks_success.jsonl` | 1 | 1 | `correct_but_slower`: 1 |
| `gemma_q4_cuda_cpp_retry_failed_tasks_all.jsonl` | 10 | 2 | `compile_fix_needed`: 7, `correct_but_slower`: 2, `correctness_fix_needed`: 1 |
| `gemma_q4_cuda_cpp_retry_failed_tasks_success.jsonl` | 2 | 2 | `correct_but_slower`: 2 |
| `gemma_q4_cuda_cpp_train_trace_v1_all.jsonl` | 14 | 5 | `correct_and_faster`: 4, `compile_fix_needed`: 9, `correct_but_slower`: 1 |
| `gemma_q4_cuda_cpp_train_trace_v1_success.jsonl` | 5 | 5 | `correct_and_faster`: 4, `correct_but_slower`: 1 |
| `gemma_q4_cuda_cpp_train_trace_v2_all.jsonl` | 15 | 2 | `compile_fix_needed`: 13, `correct_but_slower`: 2 |
| `gemma_q4_cuda_cpp_train_trace_v2_success.jsonl` | 2 | 2 | `correct_but_slower`: 2 |
| `gemma_q4_cuda_cpp_train_trace_v2_repair_all.jsonl` | 12 | 0 | `compile_fix_needed`: 12 |
| `gemma_q4_cuda_cpp_train_trace_v2_repair_success.jsonl` | 0 | 0 | none |
| `gemma_q4_cuda_cpp_mechanical_repair_v1_all.jsonl` | 40 | 26 | `mechanical_repair_correct_but_slower`: 19, `mechanical_repair_correct_and_faster`: 7, `mechanical_repair_correctness_fix_needed`: 9, `mechanical_repair_compile_fix_needed`: 5 |
| `gemma_q4_cuda_cpp_mechanical_repair_v1_success.jsonl` | 26 | 26 | `mechanical_repair_correct_but_slower`: 19, `mechanical_repair_correct_and_faster`: 7 |
| `gemma_q4_cuda_cpp_train_trace_v3_all.jsonl` | 16 | 0 | `compile_fix_needed`: 16 |
| `gemma_q4_cuda_cpp_train_trace_v3_success.jsonl` | 0 | 0 | none |
| `gemma_q4_cuda_cpp_train_trace_v3_mechanical_repair_all.jsonl` | 16 | 4 | `mechanical_repair_correct_but_slower`: 1, `mechanical_repair_correct_and_faster`: 3, `mechanical_repair_correctness_fix_needed`: 3, `mechanical_repair_compile_fix_needed`: 9 |
| `gemma_q4_cuda_cpp_train_trace_v3_mechanical_repair_success.jsonl` | 4 | 4 | `mechanical_repair_correct_but_slower`: 1, `mechanical_repair_correct_and_faster`: 3 |
| `gemma_q4_cuda_cpp_unsolved_normalized_v1_all.jsonl` | 14 | 3 | `extraction_normalized_compile_fix_needed`: 7, `extraction_normalized_correctness_fix_needed`: 4, `extraction_normalized_correct_but_slower`: 1, `extraction_normalized_correct_and_faster`: 2 |
| `gemma_q4_cuda_cpp_unsolved_normalized_v1_success.jsonl` | 3 | 3 | `extraction_normalized_correct_but_slower`: 1, `extraction_normalized_correct_and_faster`: 2 |

Current raw native CUDA clean success rows: 13.

Current mechanically repaired native CUDA success rows: 30.

Current extraction-normalized native CUDA success rows: 3.

Current curated candidate rows if repair/normalization provenance is allowed: 46.

## Triton Comparison Datasets

| Dataset | Records | Correct | Labels |
| --- | ---: | ---: | --- |
| `gemma_q4_triton_comparison_current_all.jsonl` | 7 | 2 | `runtime_fix_needed`: 3, `compile_fix_needed`: 2, `correct_and_faster`: 1, `correct_but_slower`: 1 |
| `gemma_q4_triton_comparison_current_success.jsonl` | 2 | 2 | `correct_and_faster`: 1, `correct_but_slower`: 1 |
| `gemma_q4_triton_expanded_new_tasks_all.jsonl` | 4 | 0 | `compile_fix_needed`: 3, `static_reject`: 1 |
| `gemma_q4_triton_expanded_new_tasks_success.jsonl` | 0 | 0 | none |

Older Triton files remain for history, but the current Triton comparison reports use the two current runs above.

## Fine-Tuning Dataset Gate

- Raw native CUDA clean successes: 13
- Mechanically repaired successes: 30
- Extraction-normalized successes: 3
- Curated candidate rows with repair/normalization provenance: 46
- Target before QLoRA: 50-200 clean native CUDA examples
- Status: not enough raw data; even the provenance-labeled curated set is still below 50.

Mechanical repair and extraction normalization recovered useful candidates, but those rows must stay labeled separately from strict raw Gemma generations.
