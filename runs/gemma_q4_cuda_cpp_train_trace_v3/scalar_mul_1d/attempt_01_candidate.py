import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>

__global__ void multiply_scale_kernel(const float* x, float scale, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = x[i] * scale;
  }
 }

 torch::Tensor multiply_scale(torch::Tensor x, float scale) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  const auto block_size = 256;
  dim3 dimBlock(block_size);
  dim3 dimGrid((n + block_size - 1) / block_size));
  
  multiply_scale_kernel<<<dimGrid, dimBlock>>>(
    x.data_ptr<float>(), 
    scale, 
    out.data_ptr<float>(), 
    n
  );
  return out;
 }
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor multiply_scale(torch::Tensor x, float scale);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("multiply_scale", &multiply_scale); }
"""

 ext = load_inline(
    name="multiply_scale_ext",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=None,
    extra_cuda_cflags=["-O2"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        return ext.multiply_scale(x.contiguous(), float(scale)))