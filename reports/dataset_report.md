# Dataset Report

Current datasets are local trace datasets. They are enough to prove the workflow, not enough for QLoRA.

## gemma_q4_cuda_cpp_baseline_all.jsonl

- Records: 5
- Correct: 3
- Backends: `{'cuda_cpp': 5}`
- Labels: `{'correct_and_faster': 2, 'compile_fix_needed': 2, 'correct_but_slower': 1}`

## gemma_q4_cuda_cpp_baseline_success.jsonl

- Records: 3
- Correct: 3
- Backends: `{'cuda_cpp': 3}`
- Labels: `{'correct_and_faster': 2, 'correct_but_slower': 1}`

## gemma_q4_triton_baseline_all.jsonl

- Records: 6
- Correct: 3
- Backends: `{'triton': 6}`
- Labels: `{'runtime_fix_needed': 2, 'compile_fix_needed': 1, 'correct_but_slower': 2, 'correct_and_faster': 1}`

## gemma_q4_triton_baseline_success.jsonl

- Records: 3
- Correct: 3
- Backends: `{'triton': 3}`
- Labels: `{'correct_but_slower': 2, 'correct_and_faster': 1}`

## gemma_q4_triton_comparison_current_all.jsonl

- Records: 7
- Correct: 2
- Backends: `{'triton': 7}`
- Labels: `{'runtime_fix_needed': 3, 'compile_fix_needed': 2, 'correct_and_faster': 1, 'correct_but_slower': 1}`

## gemma_q4_triton_comparison_current_success.jsonl

- Records: 2
- Correct: 2
- Backends: `{'triton': 2}`
- Labels: `{'correct_and_faster': 1, 'correct_but_slower': 1}`

## gemma_q4_cuda_cpp_expanded_new_tasks_all.jsonl

- Records: 4
- Correct: 1
- Backends: `{'cuda_cpp': 4}`
- Labels: `{'compile_fix_needed': 2, 'correctness_fix_needed': 1, 'correct_but_slower': 1}`

## gemma_q4_cuda_cpp_expanded_new_tasks_success.jsonl

- Records: 1
- Correct: 1
- Backends: `{'cuda_cpp': 1}`
- Labels: `{'correct_but_slower': 1}`

## Fine-Tuning Dataset Gate

- Native CUDA clean successes: 4
- Target before QLoRA: 50-200 clean native CUDA examples
- Status: not enough data; do not fine-tune yet.
