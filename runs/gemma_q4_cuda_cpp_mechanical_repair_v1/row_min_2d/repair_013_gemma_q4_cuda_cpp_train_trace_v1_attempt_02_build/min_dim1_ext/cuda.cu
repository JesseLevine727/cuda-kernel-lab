#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void min_dim1_kernel(const float* input, float* output, int rows) {
   int row = blockIdx.x;
   if (row < rows) {
     float min_val = 3.402823466e38f;
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
     input.data_ptr<float>(),
      output.data_ptr<float>(),
      (int)rows
   );
   
   return output;
 }
