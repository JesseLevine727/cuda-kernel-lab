#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

__global__ void add_bias_kernel(const float* x, const float* bias, float* out, int rows, int cols) {
  int row = blockIdx.x;
  int col = blockIdx.y;
  if (row < rows && col < cols) {
    int idx = row * cols + col;
    out[idx] = x[idx] + bias[col];
  }
}

torch::Tensor add_bias(torch::Tensor x, torch::Tensor bias) {
  auto out = torch::empty_like(x);
  int rows = x.size(0);
  int cols = x.size(1);
  dim3 dimBlock(16, 16);
  dim3 dimGrid(rows, cols);
  add_bias_kernel<<<dimGrid, dimBlock>>>(x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), rows, cols);
  return out;
}
