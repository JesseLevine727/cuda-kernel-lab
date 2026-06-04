import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>

__global__ void negate_kernel(const float* input, float* output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = -input[i];
  }
 }

 torch::Tensor negate(torch::Tensor input) {
  auto output = torch::empty_like(input);
  int n = input.numel();
  negate_kernel<<<{(n + 255) / 256, 256}>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);
  return output;
 }
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor negate(torch::Tensor input);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("negate", &negate); }
"""

 ext = load_inline(
    name="negate_extension",
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
        return ext.negate(x.contiguous())