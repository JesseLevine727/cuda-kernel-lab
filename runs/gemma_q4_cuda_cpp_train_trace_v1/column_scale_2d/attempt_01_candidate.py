import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_vec_kernel(const float* x, const float* weights, float* out, int rows) {
  int row = blockIdx.x;
  if (row < rows) {
    float sum = 0.0f;
    for (int i = 0; i < 256; ++i) {
      sum += x[row * 256 + i] * weights[i];
    }
    out[row] = sum;
  }
}

torch::Tensor matmul_vec(torch::Tensor x, torch::Tensor weights) {
  auto out = torch::empty_like(x);
  int rows = x.size(0);
  dim3 blocks(rows);
  dim3 threads(1);
  matmul_vec_kernel<<<blocks, threads>>>(
    x.data_ptr<float>(),
    weights.data_ptr<float>(),
    out.data_ptr<float>(),
    rows
  );
  return out;
}
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor matmul_vec(torch::Tensor x, torch::Tensor weights);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_vec", &matmul_vec);
}
"""

ext = load_inline(
    name="matmul_vec_ext",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=None,
    extra_cuda_cflags=["-O2"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().___init__()

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return ext.matmul_vec(x.contiguous(), weights.contiguous())