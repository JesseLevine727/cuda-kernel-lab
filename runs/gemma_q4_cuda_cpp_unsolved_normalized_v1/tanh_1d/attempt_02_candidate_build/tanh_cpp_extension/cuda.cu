#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <cmath>

__global__ void tanh_kernel(const float* input, float* output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = input[i];
    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    // Using std::tanh for precision and handling of edge cases
    output[i] = (float)std::tanh(x);
  }
}

torch::Tensor tanh(torch::Tensor x) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  dim3 dimBlock(1);
  dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);
  tanh_kernel<<<dimGrid.x, dimBlock.x>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
}
