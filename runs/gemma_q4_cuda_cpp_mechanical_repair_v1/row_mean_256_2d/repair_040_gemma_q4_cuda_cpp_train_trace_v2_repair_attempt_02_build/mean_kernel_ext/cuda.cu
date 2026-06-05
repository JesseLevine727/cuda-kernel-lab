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

 torch::Tensor mean_kernel_func(torch::Tensor input) {
  auto rows = input.size(0);
  auto out = torch::empty({rows}, input.options());
  dim3 dimBlock(256);
  dim3 dimGrid(rows);
  mean_kernel<<<dimGrid.x, dimBlock.x>>>(input.data_ptr<float>(), out.data_ptr<float>(), (int)rows);
  return out;
 }
