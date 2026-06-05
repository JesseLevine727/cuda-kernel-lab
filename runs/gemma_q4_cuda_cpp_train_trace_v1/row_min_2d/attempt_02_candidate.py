import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void min_dim1_kernel(const float* input, float* output, int rows) {
   int row = blockIdx.x;
   if (row < rows) {
     float min_val = FLINF;
     for (int col = 0; col < 256; ++col) {
       float val = input[row * 256 + col];
       if (val < min_val) {
         min_val = val;
       }
     }
     output[row] = min_val;
   }
 }

 torch::Tensor min_dim1(torch::Tensor input) {
   auto rows = input.size(0);
   auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
   auto output = torch::empty({rows}, options);
   
   dim3 blocks(rows);
   dim3 threads(1);
   
   min_dim1_kernel<<<blocks, threads>>>(
     const float* input.data_ptr<float>(),
      output.data_ptr<float>(),
      (int)rows
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