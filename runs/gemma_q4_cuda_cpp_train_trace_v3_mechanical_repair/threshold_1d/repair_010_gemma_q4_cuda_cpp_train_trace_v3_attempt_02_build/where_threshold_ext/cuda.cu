#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void where_threshold_kernel(const float* x, float threshold, float value, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    if (x[i] > threshold) {
      out[i] = x[i];
    } else {
      out[i] = value;
    }
  }
 }

 torch::Tensor where_threshold(torch::Tensor x, float threshold, float value) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  dim3 dimBlock(256);
  dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);
  where_threshold_kernel<<<dimGrid.x, dimBlock.x>>>(x.data_ptr<float>(), threshold, value, out.data_ptr<float>(), n);
  return out;
 }
