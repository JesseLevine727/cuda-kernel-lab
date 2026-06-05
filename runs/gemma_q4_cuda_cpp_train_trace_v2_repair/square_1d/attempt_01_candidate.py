import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>

__global__ void square_kernel(const float* x, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float val = x[i];
    out[i] = val * val;
  }
 }

 torch::Tensor square(torch::Tensor x) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  const dim3 dimBlock(256);
   const dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);
  
  square_kernel<<<dimGrid, dimBlock>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
 }
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor square(torcheres::Tensor x);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("square", &square); }
"""

 ext = load_inline(
    name="square_ext",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=None,
    extra_cuda_cflags=["-O2"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ext.square(x.contiguous())