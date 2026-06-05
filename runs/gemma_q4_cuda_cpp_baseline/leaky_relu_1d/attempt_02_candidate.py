import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, float negative_slope, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) {
     float val = input[i];
     if (val >= 0.0f) {
       output[i] = val;
     } else {
       output[i] = val * negative_slope;
     }
   }
 }

 torch::Tensor leaky_relu_cuda(torch::Tensor x, float negative_slope) {
   auto out = torch::empty_like(x);
   int n = x.numel();
   dim3 dimBlock(256);
   dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);
   
   leaky_relu_kernel<<<dimGrid.x, dimBlock.x>>>(
       x.data_ptr<float>(), 
       out.data_ptr<float>(), 
       negative_slope, 
       n);
   return out;
 }
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor leaky_relu_cuda(torch::Tensor x, float negative_slope);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("leaky_relu_cuda", &leaky_relu_cuda); }
"""

 ext = load_inline(
     name="leaky_relu_cuda_ext",
     cpp_sources=CPP_SOURCE,
     cuda_sources=CUDA_SOURCE,
     functions=None,
     extra_cuda_cflags=["-O2"],
     verbose=False,
 )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, negative_slope: float) -> torch.Tensor:
        return ext.leaky_relu_cuda(x.contiguous(), float(negative_slope)))