import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <math.h>

__global__ void tanh_kernel(const float* input, float* output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = input[i];
    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    // To avoid overflow in exp(2x), we use the identity:
    // tanh(x) = 1 - 2 / (1 + exp(-2x)) for x > 0
    // tanh(x) = -1 + 2 / (1 - exp(2x)) for x < 0
    // Or simply use the standard math.h tanh function which handles these cases.
    output[i] = tanhf(x);
  }
}

 torch::Tensor tanh(torch::Tensor x) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  dim3 dimBlock(1);
  int numBlocks = (n + dimBlock.x - 1) / dimBlock.x;
  tanh_kernel<<<numBlocks, 256>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
 }
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor tanh(torch::Tensor x);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("tanh", &tanh); }
"""

 ext = load_inline(
    name="tanh_cpp_ext",
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
        return ext.tanh(x.contiguous())