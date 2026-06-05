import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* input, float* output, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) {
     output[i] = 1.0f / (1.0f + expf(-input[i]));
   }
 }

 torch::Tensor sigmoid(torch::Tensor x) {
   auto out = torch::empty_like(x);
   int n = x.numel();
   dim3 dimBlock(256);
   dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);
   sigmoid_kernel<<<dimGrid.x, dimBlock.x>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
   return out;
 }
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor sigmoid(torch::Tensor x);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   m.def("sigmoid", &sigmoid);
 }
"""

ext = load_inline(
    name="sigmoid_ext",
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
        return ext.sigmoid(x.contiguous())