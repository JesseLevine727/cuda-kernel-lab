#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_sq_add_kernel(const float* x, const float* z, float* out, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) {
     float val = x[i] * x[i] + z[i];
     out[i] = val > 0.0f ? val : 0.0f;
   }
 }

 torch::Tensor relu_sq_add(torch::Tensor x, torch::Tensor z) {
   auto out = torch::empty_like(x);
   int n = x.numel();
   dim3 dimBlock(256);
   dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);
   relu_sq_add_kernel<<<dimGrid.x, dimBlock.x>>>(
       x.data_ptr<float>(), 
       z.data_ptr<float>(), 
       out.data_ptr<float>(), 
       n);
   return out;
 }
