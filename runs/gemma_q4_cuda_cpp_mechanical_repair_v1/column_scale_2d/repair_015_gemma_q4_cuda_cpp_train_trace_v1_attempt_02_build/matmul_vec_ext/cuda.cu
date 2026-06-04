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

torch::Tensor matmul_vec(torch::Tensor x, torch::Tensor weights) {
  auto out = torch::empty_like(x);
  int rows = x.size(0);
  dim3 dimBlock(256);
  dim3 dimGrid((rows + 255) / 256);
  
  matmul_vec_kernel<<<dimGrid.x, dimBlock.x>>>(
    x.data_ptr<float>(), 
    weights.data_ptr<float>(), 
    out.data_ptr<float>(), 
    rows
  );
  return out;
}
