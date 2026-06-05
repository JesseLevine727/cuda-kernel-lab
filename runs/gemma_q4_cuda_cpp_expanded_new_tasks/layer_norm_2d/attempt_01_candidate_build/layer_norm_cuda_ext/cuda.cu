#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void layer_norm_kernel(const float* input, float* output, float* weight, float* bias, int rows, int cols) {
  int row = blockIdx.x;
  int col = threadIdx.x;
  
  if (row < rows && col < cols) {
    float sum = 0.0f;
    float sq_sum = 0.0f;
    
    // First pass: calculate mean and variance
    // Since we need to reduce over the last dimension (cols), 
    // we can do this in a single pass if we use shared memory or 
    // just calculate locally if we assume the kernel is launched per element.
    // To keep it simple and correct for a single kernel call:
    
    // We need the sum of the row to find the mean.
    // Because we are in a grid of (rows, cols), we must reduce across cols.
    
    // Optimization: Since we need to calculate mean/var for each row, 
    // we can use a reduction or just calculate it.
    // To keep the kernel simple for a "small" kernel, we'll use a simple approach.
    
    // Actually, a standard way to do this in one kernel is to use 
    // shared memory for the reduction of the row.
  }
}

// Revised approach for simplicity and correctness:
// We will use two passes or a simple reduction. 
// Given the requirement for a "small" kernel, let's use a simple 
// reduction for mean and variance.

__global__ void layer_norm_simple_kernel(const float* input, float* output, float* weight, float* bias, int rows, int cols) {
  int row = blockIdx.x;
  int col = threadIdx.x;
  
  if (row < rows && col < cols) {
    // We need to calculate mean and variance for the row.
    // To do this efficiently in a single kernel without complex reduction,
    // we can use a block-level reduction.
    
    // However, for a "small" kernel, we can just use a simple loop 
    // if we assume the block size is large enough or use shared memory.
    
    // Let's use a simpler approach: 
    // 1. Calculate mean and variance per row.
    // 2. Apply normalization.
    
    // Since we want to avoid complex multi-pass logic in one kernel:
    // We'll use a simple loop for the reduction if the dimension is small (256).
    
    float sum = 0.0f;
    float sq_sum = 0.0f;
    
    // This part is tricky in a single thread. We need to sum the row.
    // We'll use a simple loop over the columns.
    for (int i = 0; i < cols; ++i) {
        float val = input[row * cols + i];
        sum += val;
        sq_sum += val * val;
    }
    
    float mean = sum / cols;
    float var = (sq_sum / cols) - (mean * mean);
    if (var < 1e-6f) var = 1e-6f;
    float inv_std = rsqrtf(var);
    
    float val = input[row * cols + col];
    output[row * cols + col] = (val - mean) * inv_std * weight[col] + bias[col];
  }
}

torch::Tensor layer_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
  auto output = torch::empty_like(input);
  int rows = input.size(0);
  int cols = input.size(1);
  
  dim3 dimBlock(256);
  dim3 dimGrid(rows);
  
  layer_norm_simple_kernel<<<dimGrid, dimBlock>>>(
    input.data_ptr<float>(), 
    output.data_ptr<float>(), 
    weight.data_ptr<float>(), 
    bias.data_ptr<float>(), 
    rows, 
    cols
  );
  
  return output;
}
