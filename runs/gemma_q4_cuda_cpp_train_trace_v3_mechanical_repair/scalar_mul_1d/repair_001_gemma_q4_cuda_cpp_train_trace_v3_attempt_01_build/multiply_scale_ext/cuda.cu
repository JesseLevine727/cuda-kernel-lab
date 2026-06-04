#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

__global__ void multiply_scale_kernel(const float* x, float scale, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = x[i] * scale;
  }
 }

 torch::Tensor multiply_scale(torch::Tensor x, float scale) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  const auto block_size = 256;
  dim3 dimBlock(block_size);
  dim3 dimGrid((n + block_size - 1) / block_size);
  
  multiply_scale_kernel<<<dimGrid, dimBlock>>>(
    x.data_ptr<float>(), 
    scale, 
    out.data_ptr<float>(), 
    n
  );
  return out;
 }
