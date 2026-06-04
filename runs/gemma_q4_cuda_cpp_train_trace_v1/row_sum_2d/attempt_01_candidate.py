import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_dim1_kernel(const float* input, float* output, int rows) {
  int row = blockIdx.x;
  if (row < rows) {
    float sum = 0.0f;
    for (int col = 0; col < 256; ++col) {
      sum += input[row * 256 + col];
    }
    output[row] = sum;
  }
}

torch::Tensor sum_dim1(torch::Tensor x) {
  auto rows = x.size(0);
  auto out = torch::empty({rows}, x.options());
  dim3 blocks(rows);
  dim3 threads(1);
  sum_dim1_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), rows);
  return out;
}
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor sum_dim1(torch::Tensor x);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sum_dim1", &sum_dim1);
}
"""

ext = load_inline(
    name="sum_dim1_ext",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=None,
    extra_cuda_cflags=["-O2"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ext.sum_dim1(x.contiguous())