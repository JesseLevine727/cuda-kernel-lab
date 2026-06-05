#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_dim1_kernel(const float* input, float* output, int rows) {
  int row = blockIdx.x;
  if (row < rows) {
    float sum = 0.0f;
    for (int col = 0; col < 256; ++col) {
      sum += input[row * 256 + col];
    }
    output[row] = sum;
  }
}

torch::Tensor sum_dim1(torch::Tensor x) {
  auto rows = x.size(0);
  auto out = torch::empty({rows}, x.options());
  dim3 blocks(rows);
  dim3 threads(1);
  sum_dim1_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), rows);
  return out;
}
