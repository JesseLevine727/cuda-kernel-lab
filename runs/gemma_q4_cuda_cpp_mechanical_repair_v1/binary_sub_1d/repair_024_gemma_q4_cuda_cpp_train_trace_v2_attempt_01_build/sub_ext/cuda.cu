#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

__global__ void sub_kernel(const float* x, const float* z, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = x[i] - z[i];
  }
 }

 torch::Tensor sub(torch::Tensor x, torch::Tensor z) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  dim3 dimBlock(256);
  dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);
  sub_kernel<<<dimGrid.x, dimBlock.x>>>(x.data_ptr<float>(), z.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
 }
