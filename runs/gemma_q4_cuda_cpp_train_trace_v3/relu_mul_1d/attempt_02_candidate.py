import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_mul_kernel(const float* x, const float* y, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float val = x[i];
    if (val < 0.0f) val = 0.0f;
    out[i] = val * y[i];
  }
 }

 torch::Tensor relu_mul(torch::Tensor x, torch::Tensor y) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  dim3 dimBlock(256);
  dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);
  relu_mul_kernel<<<dimGrid.x, dimBlock.x>>>(
      x.data_ptr<float>(), 
      y.data_ptr<float>(), 
      out.data_ptr<float>(), 
      n);
  return out;
 }
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor relu_mul(torch::Tensor x, torch::Tensor y);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("relu_mul", &relu_mul); }
"""

 ext = load_inline(
     name="relu_mul_ext",
     cpp_sources=CPP_SOURCE,
     cuda_sources=CUDA_SOURCE,
     functions=None,
     extra_cuda_cflags=["-O2"],
     verbose=False,
 )

 class ModelNew(nn.Module):
     def __init__(self):
         super().__init__()

     def forward(self, x, y):
         return ext.relu_mul(x.contiguous(), y.contiguous())