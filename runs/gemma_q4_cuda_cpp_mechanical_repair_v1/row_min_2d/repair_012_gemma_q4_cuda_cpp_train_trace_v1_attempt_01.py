import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void min_kernel(const float* input, float* output, int rows, int cols) {
   int row = blockIdx.x;
   int col = blockIdx.y;
   
   if (row < rows && col < cols) {
       float min_val = input[row * cols + col];
       for (int i = 1; i < cols; ++i) {
           float val = input[row * cols + (col + i)];
           if (val < min_val) {
               min_val = val;
           }
       }
       // We only need to store the result for the first column of the reduction
       // but the logic above finds the min of the row.
   }
}

// Optimized version for reduction
__global__ void min_reduction_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row < rows) {
        float min_val = 3.402823466e38f();
        for (int i = 0; i < cols; ++i) {
            float val = input[row * cols + i];
            if (val < min_val) {
                min_val = val;
            }
        }
        output[row] = min_val;
    }
}

torch::Tensor min_dim1(torch::Tensor input) {
    auto rows = input.size(0);
    auto cols = input.size(1);
    auto output = torch::empty({rows}, torch::kFloat32);
    
    dim3 blocks(rows);
    dim3 threads(1);
    
    min_reduction_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)rows,
        (int)cols
    );
    
    return output;
}
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor min_dim1(torch::Tensor input);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
     m.def("min_dim1", &min_dim1);
 }
"""

ext = load_inline(
    name="min_dim1_ext",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=None,
    extra_cuda_cflags=["-O2"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be (rows, 256)
        return ext.min_dim1(x.contiguous())