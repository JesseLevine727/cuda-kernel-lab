#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

__global__ void scale_kernel(const float* x, float scale, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = x[i] * scale;
  }
 }

 torch::Tensor scale(torch::Tensor x, float scale) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  scale_kernel<<<{(n + 255) / 256, 256}>>>(x.data_ptr<float>(), scale, out.data_ptr<float>(), n);
  return out;
 }
