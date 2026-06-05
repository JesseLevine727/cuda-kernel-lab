import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_bias_kernel(const float* x, const float* bias, float* out, int rows, int cols) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < rows && j < cols) {
    float val = x[i * cols + j] + bias[j];
    out[i * cols + j] = val > 0.0f ? val : 0.0f;
  }
 }

 torch::Tensor relu_bias(torch::Tensor x, torch::Tensor bias) {
  auto out = torch::empty_like(x);
  int rows = x.size(0);
  int cols = x.size(1);
  dim3 dimBlock(16, 16);
  dim3 dimGrid((rows + 15) / 16, (cols + 15) / 16);
  relu_bias_kernel<<<dimGrid, dimBlock>>>(x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), rows, cols);
  return out;
 }
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor relu_bias(torch::Tensor x, torch::Tensor bias);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("relu_bias", &relu_bias); }
"""

ext = load_inline(
     name="relu_bias_ext",
     cpp_sources=CPP_SOURCE,
     cuda_sources=CUDA_SOURCE,
     functions=None,
     extra_cuda_cflags=["-O2"],
     verbose=False,
 )

class ModelNew(nn.Module):
     def __init__(self):
         super().__init__()

     def forward(self, x, bias):
         return ext.relu_bias(x.contiguous(), bias.contiguous()