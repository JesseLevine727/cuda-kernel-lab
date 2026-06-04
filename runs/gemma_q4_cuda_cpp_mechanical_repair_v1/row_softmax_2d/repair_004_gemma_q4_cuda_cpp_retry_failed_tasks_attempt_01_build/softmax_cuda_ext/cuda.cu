#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    int col = blockIdx.y;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        
        float max_val = -1e38f;
        for (int i = 0; i < cols; ++i) {
            max_val = fmaxf(max_val, input[row * cols + i]);
        }
        
        float sum = 0.0f;
        for (int i = 0; i < cols; ++i) {
            float val = expf(input[row * cols + i] - max_val);
            output[idx] = val;
            sum += val;
        }
        
        output[idx] /= sum;
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int rows = x.size(0);
    int cols = x.size(1);
    
    dim3 block(16, 16);
    dim3 grid((rows + 15) / 16, (cols + 15) / 16);
    
    softmax_kernel<<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>(), rows, cols);
    return out;
}
