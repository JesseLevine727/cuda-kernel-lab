import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                               int M, int N, int K) {
  int row = blockIdx.x;
  int col = blockIdx.y;
  
  float sum = 0.0f;
  for (int i = 0; i < K; ++i) {
    sum += A[row * K + i] * B[i * N + col];
  }
  C[row * N + col] = sum;
}

 torch::Tensor matmul(torch::Tensor a, torch::Tensor b) {
  auto C = torch::empty({a.size(0), b.size(1)}, a.options());
  int M = a.size(0);
  int K = a.size(1);
  int N = b.size(1);
  
  dim3 grid(M, N);
  dim3 block(16, 16);
  
  matmul_kernel<<<GRID, BLOCK>>>(
    a.data_ptr<float>(), 
    b.data_ptr<float>(), 
    C.data_ptr<float>(), 
    M, N, K);
  return C;
}
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor matmul(torch::Tensor a, torch::Tensor b);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("matmul", &matmul); }
"""

 ext = load_inline(
    name="matmul_cpp_kernel",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=None,
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are contiguous and float32 as required by the kernel
        a_contig = a.contiguous().to(torch.float32)
        b_contig = b.contiguous().to(torch.float32)
        return ext.matmul(a_contig, b_contig)