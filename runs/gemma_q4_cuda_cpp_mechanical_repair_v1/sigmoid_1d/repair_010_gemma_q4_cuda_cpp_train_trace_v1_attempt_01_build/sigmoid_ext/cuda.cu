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
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  sigmoid_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
 }
