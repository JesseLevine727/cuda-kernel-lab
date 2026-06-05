# Completion Audit

## Completed

- Pinned Q4 model preserved and sanity checked.
- Repo-local CUDA 13.0 `nvcc` selected without driver upgrade.
- Standalone `.cu` vector-add compiles with `-arch=sm_120` and runs.
- PyTorch CUDA extension compiles with `compute_120/sm_120` and runs.
- Native CUDA evaluator backend exists and captures compile/runtime logs.
- `cuda_cpp` static gate requires actual native CUDA extension structure.
- Hand-written native CUDA candidate passes evaluator.
- Thirty-two-task reduced suite exists with dev, train-trace, and held-out split separation.
- Gemma Q4 native CUDA first-pass baseline exists across the original eight tasks.
- Gemma Q4 native CUDA targeted retry run exists for failed/harder original tasks.
- Gemma Q4 native CUDA train-trace v1, v2, and v3 runs exist.
- Gemma Q4 native CUDA focused v2 repair run exists and records failure evidence.
- Gemma Q4 native CUDA mechanical repair runs exist and recover 30 correct repaired rows.
- Gemma Q4 native CUDA extraction-normalized retry exists and recovers 3 correct normalized rows.
- Gemma Q4 Triton comparison exists across the original eight tasks.
- Pinned Q8 loads at 4k context and passes a trivial generation sanity check.
- Current trace datasets and reports exist.

## Still Open Against Full Goal

- Collect 50-200 clean native CUDA examples; current raw count is 13.
- Decide how much provenance-labeled repair/normalization data is acceptable; current repaired/normalized success count is 33.
- Add more native CUDA train-trace tasks beyond the current 32 registered tasks.
- Improve prompt-side compile-error repair, especially top-level indentation and malformed generated modules.
- Run staged Q8 context and quality comparisons if Q8 is worth pursuing.
- QLoRA only if the dataset quality gate later passes.
- Final before/after tuned comparison only if tuning happens.

Do not mark the active goal complete yet; core native CUDA support is done, but dataset scale and optional tuning/comparison gates remain open.
