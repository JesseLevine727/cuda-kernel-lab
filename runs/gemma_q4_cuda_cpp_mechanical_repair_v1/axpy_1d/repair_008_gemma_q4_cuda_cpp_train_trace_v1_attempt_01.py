import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>

__global__ void linear_kernel(const float* x, const float* y, float* out, float alpha, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
   out[i] = alpha * x[i] + y[i];
  }
 }

 torch::Tensor linear(torch::Tensor x, torch::Tensor y, float alpha) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  dim3 dimBlock(256);
  dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);
  linear_kernel<<<dimGrid.x, dimBlock.x>>>(
    x.data_ptr<float>(), 
    y.data_ptr<float>(), 
    out.data_ptr<float>(), 
    alpha, 
    n);
  return out;
 }
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor linear(torch::Tensor x, torch::Tensor y, float alpha);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("linear", &linear); }
"""

ext = load_inline(
    name="linear_ext",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=None,
    extra_cuda_cflags=["-O2"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
        return ext.linear(x.contiguous(), y.contiguous(), float(alpha))