#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

__global__ void affine_kernel(const float* x, float* out, float scale, float bias, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = x[i] * scale + bias;
  }
}

torch::Tensor affine(torch::Tensor x, double scale, double bias) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  affine_kernel<<<(n + 255) / 256, 256>>>(
      x.data_ptr<float>(), out.data_ptr<float>(), static_cast<float>(scale), static_cast<float>(bias), n);
  return out;
}
