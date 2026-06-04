#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void abs_kernel(const float* input, float* output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float val = input[i];
    output[i] = (val < 0.0f) ? -val : val;
  }
 }

 torch::Tensor abs_func(torch::Tensor x) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  abs_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
 }
