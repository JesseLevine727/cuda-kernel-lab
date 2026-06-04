#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {
  int row = blockIdx.x;
  int col = threadIdx.x;
  
  if (row < rows && col < cols) {
    float max_val = -1e38f;
    for (int i = 0; i < cols; ++i) {
      float val = input[row * cols + i];
      if (val > max_val) max_val = val;
    }
    
    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) {
      float val = expf(input[row * cols + i] - max_val);
      output[row * cols + i] = val;
      sum += val;
    }
    
    for (int i = 0; i < cols; ++i) {
      output[row * cols + i] /= sum;
    }
  }
}

 torch::Tensor softmax(torch::Tensor x) {
  auto out = torch::empty_like(x);
  int rows = x.size(0);
  int cols = x.size(1);
  dim3 dimBlock(cols);
  dim3 dimGrid(rows);
  
  softmax_kernel<<<dimGrid, dimBlock>>>(x.data_ptr<float>(), out.data_ptr<float>(), rows, cols);
  return out;
 }
