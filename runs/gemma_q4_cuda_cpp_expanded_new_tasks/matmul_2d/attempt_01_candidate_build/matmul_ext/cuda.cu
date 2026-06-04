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
    auto M = a.size(0);
    auto K = a.size(1);
    auto N = b.size(1);
    auto C = torch::empty({M, N}, a.options());
    
    dim3 threads(16, 16);
    dim3 blocks((M + 15) / 16, (N + 15) / 16);
    
    matmul_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), 
        b.data_ptr<float>(), 
        C.data_ptr<float>(), 
        (int)M, (int)N, (int)K);
    
    return C;
}
