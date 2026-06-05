#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* input, float* output, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) {
     output[i] = 1.0f / (1.0f + expf(-input[i]));
   }
 }

 torch::Tensor sigmoid(torch::Tensor x) {
   auto out = torch::empty_like(x);
   int n = x.numel();
   dim3 dimBlock(256);
   dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);
   sigmoid_kernel<<<dimGrid.x, dimBlock.x>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
   return out;
 }
