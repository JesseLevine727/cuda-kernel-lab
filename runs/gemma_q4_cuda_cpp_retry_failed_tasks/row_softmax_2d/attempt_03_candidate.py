import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    
    if (row < rows && col < cols) {
        float max_val = -1e38f;
        for (int i = 0; i < cols; ++i) {
            float val = input[row * cols + i];
            if (val > max_val) max_val = val;
        }
        
        float sum = 0.0f;
        float exp_val = expf(input[row * cols + col] - max_val);
        
        // We need to calculate the sum of exponentials for the specific row.
        // Since we want to be correct and simple, we can do a two-pass approach 
        // or a reduction. For a fixed width of 256, we can use shared memory 
        // or just a simple loop if we process the row correctly.
        
        // To ensure correctness in a single pass for the sum, we use a 
        // reduction or a simple loop over the columns for the sum.
        
        float temp_sum = 0.0f;
        for (int i = 0; i < cols; ++i) {
            temp_sum += expf(input[row * cols + i] - max_val);
        }
        
        output[row * cols + col] = exp_val / temp_sum;
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int rows = input.size(0);
    int cols = input.size(1);
    
    dim3 dimBlock(256);
    dim3 dimGrid(rows);
    
    softmax_kernel<<<dimGrid, dimBlock>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        rows, 
        cols
    );
    
    return output;
}
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor softmax_cuda(torch::Tensor input);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_cuda", &softmax_cuda);
}
"""

ext = load_inline(
    name="softmax_cuda_ext",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=None,
    extra_cuda_cflags=["-O3"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be (rows, 256)
        return ext.softmax_cuda(x.contiguous())