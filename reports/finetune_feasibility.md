# Fine-Tuning Feasibility

Decision: **do not fine-tune yet**.

Evidence:

- Native CUDA eval works with repo-local CUDA 13.0 `nvcc` and RTX 5080 `sm_120`.
- The native CUDA static gate requires actual `load_inline` CUDA/C++ candidates for `cuda_cpp`.
- Original 8-task eval: 4/8 clean first-pass, 6/8 best-known after retries.
- Train-trace v1: 5/8 solved.
- Train-trace v2: 2/8 solved.
- Focused v2 prompt repair: 0/6 solved.
- Train-trace v3: 0/8 solved raw; v3 mechanical repair solved 3/8 with 4 correct repaired rows.
- Mechanical repair total: 16/32 solved, 30 correct repaired rows.
- Extraction-normalized retry: 3/8 solved, 3 correct normalized rows.
- Best-known native CUDA coverage with provenance-labeled repaired/normalized candidates: 27/32.
- Raw native CUDA clean success traces: 13.
- Curated candidate rows with repair/normalization provenance: 46.
- Target before QLoRA: 50-200 clean native CUDA examples.
- Training must use original HF safetensors, not GGUF.
- Local 16GB VRAM makes full 12B tuning unrealistic; LoRA/QLoRA is the only plausible local route.

Next gate: collect more raw or provenance-safe native CUDA successes. Do not train until the native CUDA dataset has enough clean, diverse, provenance-safe examples.
