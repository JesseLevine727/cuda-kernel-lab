#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

__global__ void add_bias_kernel(const float* x, float bias, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = x[i] + bias;
  }
 }

 torch::Tensor add_bias(torch::Tensor x, float bias) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  add_bias_kernel<<<{(n + 255) / 256, 256}>>>(x.data_ptr<float>(), bias, out.data_ptr<float>(), n);
  return out;
 }
