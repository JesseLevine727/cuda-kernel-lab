import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>

__global__ void fused_mul_add_kernel(const float* x, const float* y, const float* z, float alpha, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = x[i] + alpha * y[i] * z[i];
  }
}

torch::Tensor fused_mul_add(torch::Tensor x, torch::Tensor y, torch::Tensor z, float alpha) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  fused_mul_add_kernel<<<blocks, threads>>>(
    x.data_ptr<float>(),
    y.data_ptr<float>(),
    z.data_ptr<float>(),
    alpha,
    out.data_ptr<float>(),
    n
  );
  return out;
}
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor fused_mul_add(torch::Tensor x, torch::Tensor y, torch::Tensor z, float alpha);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fused_mul_add", &fused_mul_add); }
"""

 ext = load_inline(
    name="fused_mul_add_ext",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=None,
    extra_cuda_cflags=["-O2"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, alpha: float) -> torch.Tensor:
        return ext.fused_mul_add(x.contiguous(), y.contiguous(), z.contiguous(), float(alpha)))