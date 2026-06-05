import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


CUDA_SOURCE = r"""
#include <torch/extension.h>

__global__ void affine_kernel(const float* x, float* out, float scale, float bias, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = x[i] * scale + bias;
  }
}

torch::Tensor affine(torch::Tensor x, double scale, double bias) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  affine_kernel<<<(n + 255) / 256, 256>>>(
      x.data_ptr<float>(), out.data_ptr<float>(), static_cast<float>(scale), static_cast<float>(bias), n);
  return out;
}
"""


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor affine(torch::Tensor x, double scale, double bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("affine", &affine);
}
"""


affine_ext = load_inline(
    name="affine_cuda_cpp_ext",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=None,
    extra_cuda_cflags=["-O2"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, scale: float, bias: float) -> torch.Tensor:
        return affine_ext.affine(x.contiguous(), scale, bias)

