# Model Pin Report

The llama.cpp server and CLI should use the pinned known-good Q4 GGUF:

```text
/home/elfo/.cache/huggingface/hub/models--ggml-org--gemma-4-12B-it-GGUF/snapshots/0f3915622134b2b6279d02f482cb12adc3d9ca3d/gemma-4-12B-it-Q4_K_M.gguf
```

The moving Hugging Face `Q4_K_M` ref previously pulled a newer snapshot that emitted repeated `<unused49>` tokens. Do not use it without a generation sanity check.