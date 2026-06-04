#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

__global__ void mul_kernel(const float* a, const float* b, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] * b[i];
  }
 }

 torch::Tensor mul(torch::Tensor a, torch::Tensor b) {
  auto out = torch::empty_like(a);
  int n = a.numel();
  dim3 dimBlock(256);
  dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);
  mul_kernel<<<dimGrid.x, dimBlock.x>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
 }
