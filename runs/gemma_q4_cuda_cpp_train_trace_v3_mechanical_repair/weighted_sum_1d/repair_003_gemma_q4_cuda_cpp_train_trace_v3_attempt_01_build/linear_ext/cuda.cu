#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

__global__ void linear_kernel(const float* x, const float* y, float* out, float alpha, float beta, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = alpha * x[i] + beta * y[i];
  }
 }

 torch::Tensor linear(torch::Tensor x, torch::Tensor y, float alpha, float beta) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  linear_kernel<<<{(n + 255) / 256, 256}>>>(x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), alpha, beta, n);
  return out;
 }
