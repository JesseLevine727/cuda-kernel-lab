#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_vec_kernel(const float* x, const float* weights, float* out, int rows) {
  int row = blockIdx.x;
  if (row < rows) {
    float sum = 0.0f;
    for (int i = 0; i < 256; ++i) {
      sum += x[row * 256 + i] * weights[i];
    }
    out[row] = sum;
  }
}

torch::Tensor matmul_vec(torch::Tensor x, torch::Tensor weights) {
  auto out = torch::empty_like(x);
  int rows = x.size(0);
  dim3 blocks(rows);
  dim3 threads(1);
  matmul_vec_kernel<<<blocks, threads>>>(
    x.data_ptr<float>(),
    weights.data_ptr<float>(),
    out.data_ptr<float>(),
    rows
  );
  return out;
}
