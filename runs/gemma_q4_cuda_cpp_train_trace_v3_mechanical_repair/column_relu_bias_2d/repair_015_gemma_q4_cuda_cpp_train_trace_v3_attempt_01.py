import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_bias_kernel(const float* x, const float* bias, float* out, int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row = idx / cols;
  int col = idx % cols;
  if (row < rows && col < cols) {
    float val = x[row * cols + col] + bias[col];
    out[row * cols + col] = val > 0.0f ? val : 0.0f;
  }
}

torch::Tensor relu_bias(torch::Tensor x, torch::Tensor bias) {
  auto out = torch::empty_like(x);
  int rows = x.size(0);
  int cols = x.size(1);
  int n = rows * cols;
  dim3 dimBlock(256);
  dim3 dimGrid((n + 255) / 256);
  relu_bias_kernel<<<dimGrid, dimBlock>>>(x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), rows, cols);
  return out;
}
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor relu_added_bias(torch::Tensor x, torch::Tensor bias);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("relu_added_bias", &relu_added_bias); }
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
        return ext.relu_added_bias(x.contiguous(), bias.contiguous()