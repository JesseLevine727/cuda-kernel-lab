import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>

__global__ void fused_add_mul_kernel(const float* x, const float* y, const float* z, float alpha, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = x[i] + alpha * y[i] * z[i];
  }
 }

 torch::Tensor fused_add_mul(torch::Tensor x, torch::Tensor y, torch::Tensor z, float alpha) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  fused_add_mul_kernel<<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), alpha, out.data_ptr<float>(), n);
  return out;
 }
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor fused_add_mul(torch::Tensor x, torch::Tensor y, torch::Tensor z, float alpha);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fused_add_mul", &fused_add_mul); }
"""

ext = load_inline(
     name="fused_add_mul_ext",
     cpp_sources=CPP_SOURCE,
     cuda_sources=CUDA_SOURCE,
     functions=None,
     extra_cuda_cflags=["-O2"],
     verbose=False,
 )

class ModelNew(nn.Module):
     def __init__(self):
         super().__init__()
 
     def forward(self, x, y, z, alpha):
         return ext.fused_add_mul(x.contiguous(), y.contiguous(), z.contiguous(), float(alpha))