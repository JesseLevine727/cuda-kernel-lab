#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

__global__ void square_kernel(const float* input, float* output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float val = input[i];
    output[i] = val * val;
  }
 }

 torch::Tensor square(torch::Tensor x) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  const dim3 dimBlock(256);
   const dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);
  
  square_kernel<<<dimGrid, dimBlock>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
 }
