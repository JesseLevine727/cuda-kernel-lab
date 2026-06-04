import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>

__global__ void scale_kernel(const float* x, float scale, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = x[i] * scale;
  }
 }

 torch::Tensor scale(torch::Tensor x, float scale) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  scale_kernel<<<{(n + 255) / 256, 256}>>>(x.data_ptr<float>(), scale, out.data_ptr<float>(), n);
  return out;
 }
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor scale(torch::Tensor x, float scale);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("scale", &scale); }
"""

ext = load_inline(
    name="scale_kernel_ext",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=None,
    extra_cuda_cflags=["-O2"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        return ext.scale(x.contiguous(), float(scale))