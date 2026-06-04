#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

__global__ void negate_kernel(const float* input, float* output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = -input[i];
  }
 }

 torch::Tensor negate(torch::Tensor input) {
  auto output = torch::empty_like(input);
  int n = input.numel();
  dim3 dimBlock(256);
  dim3 dimGrid((n + 255) / 256);
  negate_kernel<<<dimGrid, dimBlock>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);
  return output;
 }
