#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

__global__ void linear_kernel(const float* x, float scale, float bias, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = x[i] * scale + bias;
  }
 }

 torch::Tensor linear(torch::Tensor x, float scale, float bias) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  int threads = 256;
  dim3 blocks((n + threads - 1) / threads);
  linear_kernel<<<blocks, threads>>>(x.data_ptr<float>(), scale, bias, out.data_ptr<float>(), n);
  return out;
 }
