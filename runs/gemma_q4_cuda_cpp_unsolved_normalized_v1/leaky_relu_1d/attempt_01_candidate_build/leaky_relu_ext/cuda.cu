#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

__global__ void leaky_relu_kernel(const float* x, float* out, float negative_slope, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float val = x[i];
    out[i] = (val > 0.0f) ? val : val * negative_slope;
  }
 }

 torch::Tensor leaky_relu(torch::Tensor x, float negative_slope) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  dim3 dimBlock(256);
  dim3 dimGrid((n + 255) / 256);
  leaky_relu_kernel<<<dimGrid.x, dimBlock.x>>>(x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n);
  return out;
 }
