#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
  int row = blockIdx.x;
  int col = blockIdx.y;
  
  for (int i = 0; i < K; ++i) {
    float val = 0.0f;
    for (int k = 0; k < K; ++k) {
      val += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] += val;
  }
 }

 torch::Tensor matmul(torch::Tensor a, torch::Tensor b) {
  auto C = torch::empty_like(a);
  int M = a.size(0);
  int K = a.size(1);
  int N = b.size(1);
  
  // Initialize C to zero
  at::fill_(C, 0);

  dim3 block(16, 16);
  dim3 grid(M, N);

  matmul_kernel<<<GRID.x, GRID.y, block.x, block.y>>>(
    a.data_ptr<float>(), 
    b.data_ptr<float>(), 
    C.data_ptr<float>(), 
    M, K, N);
  return C;
 }
