from __future__ import annotations

import os
import traceback

import torch
from torch.utils.cpp_extension import load_inline


def main() -> int:
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    os.environ.setdefault("MAX_JOBS", "1")

    cuda_source = r"""
#include <torch/extension.h>

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

torch::Tensor add(torch::Tensor a, torch::Tensor b) {
  auto out = torch::empty_like(a);
  int n = a.numel();
  add_kernel<<<(n + 255) / 256, 256>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
}
"""

    cpp_source = r"""
#include <torch/extension.h>

torch::Tensor add(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", &add);
}
"""

    try:
        mod = load_inline(
            name="native_cuda_vector_add_smoke",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=None,
            extra_cuda_cflags=["-O2"],
            verbose=True,
        )
        a = torch.randn(4096, device="cuda")
        b = torch.randn(4096, device="cuda")
        out = mod.add(a, b)
        torch.cuda.synchronize()
        if not torch.allclose(out, a + b):
            print("torch extension vector add mismatch")
            return 2
        print("torch extension vector add ok")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

