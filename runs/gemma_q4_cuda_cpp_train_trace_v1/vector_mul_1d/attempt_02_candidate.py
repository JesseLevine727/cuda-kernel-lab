import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>

__global__ void mul_kernel(const float* a, const float* b, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] * b[i];
  }
 }

 torch::Tensor mul(torch::Tensor a, torch::Tensor b) {
  auto out = torch::empty_like(a);
  int n = a.numel();
  dim3 dimBlock(256);
  dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);
  mul_kernel<<<dimGrid.x, dimBlock.x>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
 }
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor mul(torch::Tensor a, torch::Tensor b);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("mul", &mul); }
"""

ext = load_inline(
    name="mul_kernel_ext",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=None,
    extra_cuda_cflags=["-O2"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return ext.mul(x.contiguous(), z.contiguous())