# Completion Audit

Goal source: `goal.md`

## Completed Requirements

- Verify CUDA/toolchain/model environment: `cuda_kernel_lab/reports/environment.json` and `environment_report.md`.
- Clone official KernelBench: `external/KernelBench` at commit `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`.
- Confirm benchmark scope: local reduced KernelBench-style Level 1/2 tasks using the KernelBench-supported Triton backend.
- Build repeatable eval harness: `cuda_kernel_lab/tasks.py`, `prompts.py`, `run_gemma_eval.py`, `kernel_eval.py`.
- Feed PyTorch reference tasks to Gemma: prompts now include each task's PyTorch reference forward code.
- Capture generated code and traces: `cuda_kernel_lab/runs/gemma_q4_triton_reference_shapes_pinned/`.
- Compile/import/run candidates in subprocesses: `cuda_kernel_lab/kernel_eval.py`.
- Check correctness across multiple seeds and input shapes: `tasks.py` varies vector lengths and row counts by seed.
- Benchmark speed against PyTorch references: saved in each attempt's evaluator result.
- Record failures and retries: saved in `results.json` and summarized in `baseline_report.md`.
- Create trace datasets: `gemma_q4_triton_reference_shapes_pinned_all.jsonl` and `gemma_q4_triton_reference_shapes_pinned_success.jsonl`.
- Test prompt/retry loop before fine-tuning: two-attempt bounded run with feedback.
- Attempt Q8 comparison if feasible: `q8_attempt.json`; Q8 server exited during load, then Q4 was restored.
- Decide whether QLoRA is worthwhile now: `finetune_feasibility.md` says no, because only two correct traces exist and native CUDA extension evaluation is blocked by the CUDA 12.0 `nvcc` toolchain.
- Restore local serving path: Q4 server is pinned to the known-good cached GGUF snapshot in `run-server.sh` and `run-cli.sh`.

## Conditional Requirements

- Run LoRA/QLoRA fine-tune: not performed because the feasibility gate failed.
- Export tuned adapter/GGUF: not applicable because no adapter was trained.
- Baseline vs tuned comparison: intentionally deferred; `final_report.md` records the baseline reference point and the reason no after-model exists.

## Current Baseline

- Run: `gemma_q4_triton_reference_shapes_pinned`
- Correctness rate (`fast_0`): 0.500
- Faster-than-PyTorch rate (`fast_1`): 0.250
- Correct traces: 2
- Best speedup: 1.009199648419306

## Final State

The local workflow is built and verified. The evidence does not justify training Gemma 4 12B yet; the next gate is collecting 50-200 clean traces and/or installing CUDA 13 toolkit support for native CUDA extension evaluation.

