import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void max_kernel(const float* input, float* output, int rows, int cols) {
   int row = blockIdx.x;
   int thread = threadIdx.x;
   int width = 512;
   
   if (row < rows) {
       float max_val = -1e38f; // Smallest float
       for (int i = 0; i < width; ++i) {
           float val = input[row * width + i];
           if (val > max_val) {
               max_val = val;
           }
       }
       output[row] = max_val;
   }
}

torch::Tensor max_dim1(torch::Tensor x) {
   auto rows = x.size(0);
   auto cols = x.size(1);
   auto out = torch::empty({rows}, x.options());
   
   dim3 dimBlock(256);
   dim3 dimGrid(rows);
   
   max_kernel<<<dimGrid, dimBlock>>>(
       x.data_ptr<float>(), 
       out.data_ptr<float>(), 
       (int)rows, 
       (int)cols
   );
   return out;
}
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor max_dim1(torch::Tensor x);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("max_dim1", &max_dim1);
}
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
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be (rows, 512)
        return ext.max_dim1(x.contiguous())