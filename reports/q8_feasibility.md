# Q8 Feasibility

- Model: `gemma-4-12b-it-q8_0`
- GGUF: `/home/elfo/.cache/huggingface/hub/models--ggml-org--gemma-4-12B-it-GGUF/snapshots/0f3915622134b2b6279d02f482cb12adc3d9ca3d/gemma-4-12B-it-Q8_0.gguf`
- Status: `feasible_at_4k_context`
- Context tested: 4096
- KV cache type: `q8_0`
- Result: `/v1/models` returned `gemma-4-12b-it-q8_0`, and chat completion for `Reply exactly: hello` returned `hello`.
- Q4 restore: passed; `/v1/models` returned `gemma-4-12b-it-q4_k_m`, and the same `hello` sanity check passed.
- Raw attempt record: `cuda_kernel_lab/reports/q8_attempt.json`
- Server log: `logs/q8-pinned-feasibility.log`

This only proves that the pinned Q8 GGUF can load and generate at a small context. It does not prove Q8 quality, long-context stability, or kernel-generation performance.

Q8 uses about 12.1 GiB of GPU model buffer in this run, versus about 7.4 GiB for Q4. On a 16GB RTX 5080, long contexts such as 64k or 128k need staged feasibility tests because KV/cache and compute buffers can exhaust the remaining VRAM.
