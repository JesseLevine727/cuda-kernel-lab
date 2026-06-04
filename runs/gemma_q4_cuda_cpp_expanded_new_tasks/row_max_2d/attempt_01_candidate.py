import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void max_kernel(const float* input, float* output, int rows, int cols) {
   int row = blockIdx.x;
   int col = blockIdx.y;
   
   if (row < rows && col < cols) {
       float max_val = -1e38f; // Roughly -infinity for float32
       // Note: In a real production environment, we'd use a reduction 
       // but for a fixed width of 512, we can iterate or use shared memory.
       // Since the task is a simple max over dim=1, we iterate over the columns.
       for (int i = 0; i < cols; ++i) {
           float val = input[row * cols + i];
           if (val > max_val) {
               max_val = val;
           }
       }
       // We only need to store the result for each row.
       // However, the kernel is launched over the 2D grid of (rows, cols).
       // To handle the reduction correctly in a simple kernel, we can 
       // use a simple approach or a more complex reduction.
       // Given the requirement for a simple kernel, we use a simple loop.
   }
}

// Since the requirement is max(x, dim=1), we can optimize the kernel 
// to process one row per block or use a simple reduction.
// To keep it simple and correct for the specific shape (rows, 512):
__global__ void max_dim1_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row < rows) {
        float max_val = -1e38f;
        for (int i = 0; i < cols; ++i) {
            float val = input[row * cols + i];
            if (val > max_val) max_val = val;
        }
        output[row] = max_val;
    }
}

torch::Tensor max_dim1(torch::Tensor input) {
    auto rows = input.size(0);
    auto cols = input.size(1);
    auto output = torch::empty({rows}, input.options());
    
    dim3 dimBlock(16, 1, 1);
    dim3 dimGrid(rows, 1, 1);
    
    max_dim1_kernel<<<dimGrid, dimBlock>>>(
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
torch::Tensor max_dim1(torch::Tensor input);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("max_dim1", &max_dim1); }
"""

 ext = load_inline(
    name="max_dim1_ext",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=None,
    extra_cuda_cflags=["-O2"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().___init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be (rows, 512)
        return ext.max_dim1(x.contiguous())