#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_kernel(const float* input, float* output, int rows) {
  int row = blockIdx.x;
  if (row < rows) {
    float sum = 0.0f;
    for (int col = 0; col < 256; ++col) {
      sum += input[row * 256 + col];
    }
    output[row] = sum / 256.0f;
  }
}

torch::Tensor mean_dim1(torch::Tensor x) {
  auto rows = x.size(0);
  auto out = torch::empty({rows}, x.options());
  int threads = 256;
  dim3 blocks(rows);
  mean_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), rows);
  return out;
}
