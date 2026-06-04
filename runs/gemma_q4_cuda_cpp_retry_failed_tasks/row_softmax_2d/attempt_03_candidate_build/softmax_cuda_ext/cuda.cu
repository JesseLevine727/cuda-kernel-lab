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
        float exp_val = expf(input[row * cols + col] - max_val);
        
        // We need to calculate the sum of exponentials for the specific row.
        // Since we want to be correct and simple, we can do a two-pass approach 
        // or a reduction. For a fixed width of 256, we can use shared memory 
        // or just a simple loop if we process the row correctly.
        
        // To ensure correctness in a single pass for the sum, we use a 
        // reduction or a simple loop over the columns for the sum.
        
        float temp_sum = 0.0f;
        for (int i = 0; i < cols; ++i) {
            temp_sum += expf(input[row * cols + i] - max_val);
        }
        
        output[row * cols + col] = exp_val / temp_sum;
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int rows = input.size(0);
    int cols = input.size(1);
    
    dim3 dimBlock(256);
    dim3 dimGrid(rows);
    
    softmax_kernel<<<dimGrid, dimBlock>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        rows, 
        cols
    );
    
    return output;
}
