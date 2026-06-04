import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_kernel(const float* input, float* output, int rows, int cols) {
  int row = blockIdx.x;
  if (row < rows) {
    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) {
      sum += input[row * cols + i];
    }
    output[row] = sum / static_cast<float>(cols);
  }
}

torch::Tensor mean_dim1(torch::Tensor x) {
  auto rows = x.size(0);
  auto cols = x.size(1);
  auto out = torch::empty({rows}, x.options());
  
  dim3 blocks(rows);
  dim3 threads(1);
  
  mean_kernel<<<blocks, threads>>>(
    x.data_ptr<float>(), 
    out.data_ptr<float>(), 
    (int)rows, 
    (int)cols
  );
  return out;
}
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor mean_dim1(torch::Tensor x);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mean_dim1", &mean_dim1);
 }
"""

ext = load_inline(
    name="mean_dim1_ext",
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
        # x is expected to be (rows, 512)
        return ext.mean_dim1(x.contiguous())