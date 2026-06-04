import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_sq_add_kernel(const float* x, const float* z, float* out, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) {
     float val = x[i] * x[i] + z[i];
     out[i] = val > 0.0f ? val : 0.0f;
   }
 }

 torch::Tensor relu_sq_add(torch::Tensor x, torch::Tensor z) {
   auto out = torch::empty_like(x);
   int n = x.numel();
   dim3 dimBlock(256);
   dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);
   relu_sq_added_kernel<<<dimGrid.x, dimBlock.x>>>(
       x.data_ptr<float>(), 
       z.data_ptr<float>(), 
       out.data_ptr<float>(), 
       n);
   return out;
 }
"""

# Note: The kernel name in the string must match the declaration in CPP_SOURCE
# Correcting the typo in the kernel name for consistency.
CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_sq_add_kernel(const float* x, const float* z, float* out, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) {
     float val = x[i] * x[i] + z[i];
     out[i] = val > 0.0f ? val : 0.0f;
   }
 }

 torch::Tensor relu_sq_add(torch::Tensor x, torch::Tensor z) {
   auto out = torch::empty_like(x);
   int n = x.numel();
   dim3 dimBlock(256);
   dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);
   relu_sq_add_kernel<<<dimGrid.x, dimBlock.x>>>(
       x.data_ptr<float>(), 
       z.data_ptr<float>(), 
       out.data_ptr<float>(), 
       n);
   return out;
 }
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor relu_sq_add(torch::Tensor x, torch::Tensor z);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("relu_sq_add", &relu_sq_add); }
"""

ext = load_inline(
    name="relu_sq_add_ext",
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
        return ext.relu_sq_add(x.contiguous(), z.contiguous())