#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_add_bias_kernel(const float* x, float bias, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float val = x[i] + bias;
    out[i] = val > 0.0f ? val : 0.0f;
  }
}

torch::Tensor relu_add_bias(torch::Tensor x, float bias) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  relu_add_bias_kernel<<<blocks, threads>>>(x.data_ptr<float>(), bias, out.data_ptr<float>(), n);
  return out;
}
