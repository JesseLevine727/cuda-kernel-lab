import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mean_kernel(x, out, num_cols: tl.constexpr, BLOCK: tl.constexpr):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # The input is (rows, 512). We iterate over the 512 columns.
    # Since 512 is a multiple of 32, we can use a fixed block size or a loop.
    # To keep it simple and correct for the specific task (dim=1), 
    # we process the columns in chunks of BLOCK.
    
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    
    for i in range(0, num_cols, BLOCK):
        offs = tl.arange(0, BLOCK)
        mask = offs < (num_cols - i)
        
        # Load values from the current chunk of the row
        vals = tl.load(x + row_idx * 512 + i + offs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0) # Summing over the local block
        
    # The mean is the sum divided by the total number of elements in that dimension
    # Since the task specifies dim=1 and width is 511, we divide by 512.
    # However, to be robust, we use the provided num_cols.
    res = acc[0] / num_cols
    
    # Store the result in the output tensor (which is 1D)
    tl.store(out + row_idx, res)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (rows, 512)
        rows = x.shape[0]
        num_cols = x.shape[1]
        
        out = torch.empty((rows,), device=x.device, dtype=x.dtype)
        
        # We use a block size of 128 for the inner loop of the mean calculation
        # Since 512 is a multiple of 128.
        BLOCK = 128
        
        grid = lambda meta: (rows,) if rows > 0 else (1,))
        
        mean_kernel[grid](
            x, 
            out, 
            num_cols, 
            BLOCK=BLOCK
        )
        
        return out