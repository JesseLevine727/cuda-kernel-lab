#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

__global__ void matmul_vec_kernel(const float* x, const float* weights, float* out, int rows) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows) {
    float sum = 0.0f;
    for (int j = 0; j < 256; ++j) {
      sum += x[i * 256 + j] * weights[j];
    }
    out[i] = sum;
  }
}

// Note: The task asks for out = x * weights where x is (rows, 256) and weights is (256,).
// In PyTorch, x * weights where weights is (256,) broadcasts weights across the rows of x.
// This means each element (i, j) of the output is x[i, j] * weights[j].

__global__ void elementwise_mul_kernel(const float* x, const float* weights, float* out, int rows) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * 256) {
    int row = i / 256;
    int col = i % 256;
    out[i] = x[i] * weights[col];
  }
}

torch::Tensor matmul_vec_op(torch::Tensor x, torch::Tensor weights) {
  auto out = torch::empty_like(x);
  int rows = x.size(0);
  int cols = x.size(1);
  int total = rows * cols;
  
  dim3 dimBlock(256);
  dim3 dimGrid((total + 255) / 256);
  
  elementwise_mul_kernel<<<dimGrid, dimBlock>>>(
    x.data_ptr<float>(), 
    weights.data_ptr<float>(), 
    out.data_ptr<float>(), 
    total);
  return out;
}
