# Fine-Tuning Feasibility

Decision: **do not fine-tune yet**.

Evidence:

- Native CUDA eval works and has an eight-task reduced suite baseline.
- Native CUDA solved 4/8; `fast_1` is 0.250.
- Native CUDA clean successes: 4.
- Target before QLoRA: 50-200 clean native CUDA examples.
- Existing held-out split is still small and should be expanded.
- Training must use original HF safetensors, not GGUF.

Next gate: add more tasks, improve CUDA prompts/retries, collect clean traces, then reassess.