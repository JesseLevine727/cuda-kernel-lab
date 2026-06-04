#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

__global__ void linear_kernel(const float* x, const float* y, float* out, float alpha, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
   out[i] = alpha * x[i] + y[i];
  }
 }

 torch::Tensor linear(torch::Tensor x, torch::Tensor y, float alpha) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  dim3 dimBlock(256);
  dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);
  linear_kernel<<<dimGrid.x, dimBlock.x>>>(
    x.data_ptr<float>(), 
    y.data_ptr<float>(), 
    out.data_ptr<float>(), 
    alpha, 
    n);
  return out;
 }
