#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

__global__ void add_bias_kernel(const float* x, const float* bias, float* out, int rows, int cols) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < rows && j < cols) {
   out[i * cols + j] = x[i * cols + j] + bias[j];
  }
 }

 torch::Tensor add_bias(torch::Tensor x, torch::Tensor bias) {
  auto out = torch::empty_like(x);
  int rows = x.size(0);
  int cols = x.size(1);
  dim3 dimBlock(16, 16);
  dim3 dimGrid((rows + 15) / 16, (cols + 15) / 16);
  add_bias_kernel<<<dimGrid, dimBlock>>>(x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), rows, cols);
  return out;
 }
