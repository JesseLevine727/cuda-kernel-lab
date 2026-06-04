#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

__global__ void linear_kernel(const float* x, const float* weights, const float* bias, float* out, int rows) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows) {
    for (int j = 0; j < 256; ++j) {
      out[i * 256 + j] = x[i * 256 + j] * weights[j] + bias[j];
    }
  }
 }

 torch::Tensor linear(torch::Tensor x, torch::Tensor weights, torch::Tensor bias) {
  auto out = torch::empty_like(x);
  int rows = x.size(0);
  linear_kernel<<<{(rows + 255) / 256, 256}>>>(x.data_ptr<float>(), weights.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), rows);
  return out;
 }
