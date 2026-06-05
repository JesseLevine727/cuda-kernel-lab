#include <cstdio>
#include <cuda_runtime.h>

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

int main() {
  constexpr int n = 1024;
  constexpr size_t bytes = n * sizeof(float);

  float* h_a = new float[n];
  float* h_b = new float[n];
  float* h_out = new float[n];

  for (int i = 0; i < n; ++i) {
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(2 * i);
  }

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_out, bytes);

  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  add_kernel<<<(n + 255) / 256, 256>>>(d_a, d_b, d_out, n);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::printf("cuda error: %s\n", cudaGetErrorString(err));
    return 1;
  }

  cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

  for (int i = 0; i < n; ++i) {
    float expected = h_a[i] + h_b[i];
    if (h_out[i] != expected) {
      std::printf("mismatch at %d: got %.1f expected %.1f\n", i, h_out[i], expected);
      return 2;
    }
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
  delete[] h_a;
  delete[] h_b;
  delete[] h_out;

  std::printf("native cuda vector add ok\n");
  return 0;
}

