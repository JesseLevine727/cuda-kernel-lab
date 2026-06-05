# Final Evaluation Report

Status: significant progress, but the active goal remains open.

## Native CUDA Q4

Original 8-task clean first-pass baseline:

- Solved: 4/8
- Correctness (`fast_0`): 0.500
- Faster-than-PyTorch (`fast_1`): 0.250
- Compile rate: 0.556
- Attempts: 9
- Average speedup among correct: 1.05377228805201
- Best speedup: 1.672956043148107

Original 8-task best-known after targeted retries:

- Solved: 6/8
- Correctness (`best-known fast_0`): 0.750
- Faster-than-PyTorch (`best-known fast_1`): 0.250

Train-trace expansion:

- v1: 5/8 solved, `fast_1` 0.500, 14 attempts.
- v2: 2/8 solved, `fast_1` 0.000, 15 attempts.
- v2 focused prompt repair: 0/6 solved, 12 attempts.
- v3: 0/8 solved raw, 16 attempts.
- mechanical repair total: 16/32 solved, 30 correct repaired rows, 56 attempts.
- extraction-normalized retry: 3/8 solved, 3 correct normalized rows, 14 attempts.

Current best-known across 32 registered tasks:

- Raw solved: 13/32
- Extraction-normalized solved: 3/32
- Best-known with provenance-labeled repaired/normalized candidates: 27/32
- Raw faster-than-PyTorch rows: 6
- Raw native CUDA clean success traces: 13
- Mechanically repaired success traces: 30
- Extraction-normalized success traces: 3
- Curated candidate rows with repair/normalization provenance: 46
- Unresolved tasks after raw plus repair/normalization: `column_mul_add_2d`, `column_relu_bias_2d`, `matmul_2d`, `tanh_1d`, `weighted_sum_1d`

## Triton Q4 Comparison

- Solved: 2/8
- Correctness (`fast_0`): 0.250
- Faster-than-PyTorch (`fast_1`): 0.125
- Compile rate: 0.455
- Attempts: 11
- Average speedup among correct: 0.8389367334885676
- Best speedup: 1.0761904614830013

## Q8 Feasibility

Pinned Q8 loaded successfully at a 4k context and passed a trivial `hello` chat-completion sanity check. Q4 was restored and sanity-checked afterward.

This is a load/generation feasibility result only. It does not prove Q8 quality, long-context stability, or Q8 kernel-generation performance.

## Decision

Native CUDA support is real, and provenance-labeled repair/normalization recovers many malformed candidates. Fine-tuning is still premature because there are only 13 strict raw clean native CUDA success traces and 46 total curated rows.
