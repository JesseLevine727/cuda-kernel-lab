#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

__global__ void axpy_kernel(const float* x, const float* y, float* out, float alpha, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = alpha * x[i] + y[i];
  }
 }

 torch::Tensor axpy(torch::Tensor x, torch::Tensor y, float alpha) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  int threads = 256;
  dim3 blocks((n + threads - 1) / threads);
  axpy_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), alpha, n);
  return out;
 }
