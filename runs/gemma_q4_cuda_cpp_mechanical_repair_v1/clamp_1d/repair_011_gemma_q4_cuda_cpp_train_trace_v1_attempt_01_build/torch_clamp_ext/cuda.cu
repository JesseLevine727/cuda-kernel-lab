#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

__global__ void clamp_kernel(const float* input, float* output, float min_val, float max_val, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float val = input[i];
    if (val < min_val) val = min_val;
    if (val > max_val) val = max_val;
    output[i] = val;
  }
 }

 torch::Tensor clamp(torch::Tensor x, float min_val, float max_val) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  dim3 dimBlock(256);
  dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);
  
  clamp_kernel<<<dimGrid.x, dimBlock.x>>>(
    x.data_ptr<float>(), 
    out.data_ptr<float>(), 
    min_val, 
    max_val, 
    n
  );
  return out;
 }
