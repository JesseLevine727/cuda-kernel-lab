import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>

__global__ void add_bias_kernel(const float* x, float bias, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = x[i] + bias;
  }
 }

 torch::Tensor add_bias(torch::Tensor x, float bias) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  const auto block_size = 256;
  dim3 dimBlock(block_size);
  dim3 dimGrid((n + block_size - 1) / block_size);
  
  add_bias_kernel<<<dimGrid, dimBlock>>>(x.data_ptr<float>(), bias, out.data_ptr<float>(), n);
  return out;
 }
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor add_bias(torch::Tensor x, float bias);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("add_bias", &add_bias); }
"""

 ext = load_inline(
    name="add_bias_ext",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=None,
    extra_cuda_cflags=["-O2"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, bias: float) -> torch.Tensor:
        return ext.add_bias(x.contiguous(), bias)