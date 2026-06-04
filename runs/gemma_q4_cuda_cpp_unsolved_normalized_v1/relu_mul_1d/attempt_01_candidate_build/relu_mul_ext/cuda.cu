#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_mul_kernel(const float* x, const float* y, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float val_x = x[i];
    if (val_x < 0.0f) val_x = 0.0f;
    out[i] = val_x * y[i];
  }
}

torch::Tensor relu_mul(torch::Tensor x, torch::Tensor y) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  dim3 block(256);
  dim3 grid((n + 255) / 256);
  relu_mul_kernel<<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
}
