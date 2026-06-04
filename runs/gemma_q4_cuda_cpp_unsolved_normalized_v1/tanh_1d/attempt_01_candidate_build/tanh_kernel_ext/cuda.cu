#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <cmath>

__global__ void tanh_kernel(const float* input, float* output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = tanhf(input[i]);
  }
}

torch::Tensor tanh_func(torch::Tensor x) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  dim3 dimBlock(256);
  dim3 dimGrid((n + 255) / 256);
  tanh_kernel<<<dimGrid.x, dimBlock.x>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
}
