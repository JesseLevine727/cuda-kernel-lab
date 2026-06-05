#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

__global__ void fused_add_mul_kernel(const float* x, const float* y, const float* z, float alpha, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = x[i] + alpha * y[i] * z[i];
  }
 }

 torch::Tensor fused_add_mul(torch::Tensor x, torch::Tensor y, torch::Tensor z, float alpha) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  fused_add_mul_kernel<<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), alpha, out.data_ptr<float>(), n);
  return out;
 }
