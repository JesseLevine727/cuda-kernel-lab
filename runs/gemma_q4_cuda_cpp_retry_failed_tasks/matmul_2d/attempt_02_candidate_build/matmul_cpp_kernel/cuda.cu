#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
  int row = blockIdx.x;
  int col = blockIdx.y;
  
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
      sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
  }
 }

 torch::Tensor matmul(torch::Tensor a, torch::Tensor b) {
  auto C = torch::empty_like(a);
  int M = a.size(0);
  int K = a.size(1);
  int N = b.size(1);
  
  dim3 grid(M, N);
  dim3 block(16, 16);
  
  matmul_kernel<<<GRID, block>>>(
    a.data_ptr<float>(), 
    b.data_ptr<float>(), 
    C.data_ptr<float>(), 
    M, N, K);
  return C;
 }
