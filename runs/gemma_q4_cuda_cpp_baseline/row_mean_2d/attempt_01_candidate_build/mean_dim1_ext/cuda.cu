#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_kernel(const float* input, float* output, int rows, int cols) {
  int row = blockIdx.x;
  if (row < rows) {
    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) {
      sum += input[row * cols + i];
    }
    output[row] = sum / static_cast<float>(cols);
  }
}

torch::Tensor mean_dim1(torch::Tensor x) {
  auto rows = x.size(0);
  auto cols = x.size(1);
  auto out = torch::empty({rows}, x.options());
  
  dim3 blocks(rows);
  dim3 threads(1);
  
  mean_kernel<<<blocks, threads>>>(
    x.data_ptr<float>(), 
    out.data_ptr<float>(), 
    (int)rows, 
    (int)cols
  );
  return out;
}
