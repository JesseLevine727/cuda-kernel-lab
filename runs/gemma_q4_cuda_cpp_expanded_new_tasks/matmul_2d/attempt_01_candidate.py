import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.x;
    int col = blockIdx.y;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul(torch::Tensor a, torch::Tensor b) {
    auto M = a.size(0);
    auto K = a.size(1);
    auto N = b.size(1);
    auto C = torch::empty({M, N}, a.options());
    
    dim3 threads(16, 16);
    dim3 blocks((M + 15) / 16, (N + 15) / 16);
    
    matmul_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), 
        b.data_ptr<float>(), 
        C.data_ptr<float>(), 
        (int)M, (int)N, (int)K);
    
    return C;
}
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor matmul(torchെയTensor a, torch::Tensor b);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("matmul", &matmul); }
"""

ext = load_inline(
    name="matmul_ext",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=None,
    extra_cuda_cflags=["-O2"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return ext.matmul(a.contiguous(), b.contiguous())