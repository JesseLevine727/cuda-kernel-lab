# Q8 Feasibility

- Model: `gemma-4-12b-it-q8_0`
- Status: `not_feasible_in_live_run`
- Reason: The temporary Q8 server did not become ready within the readiness window and the process exited during model load. The Q4 server was restored afterward.

Q8 should only be retried with a pinned known-good GGUF and generation sanity checks before running benchmarks.