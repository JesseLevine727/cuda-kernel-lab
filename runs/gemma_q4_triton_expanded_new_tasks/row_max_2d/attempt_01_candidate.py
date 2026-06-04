import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def max_kernel(x_ptr, out_ptr, num_rows: tl.constexpr, BLOCK: tl.constexpr):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # The dimension size is fixed at 512
    cols = 512
    offs = tl.arange(0, BLOCK)
    mask = offs < cols
    
    # Load the row data
    # Since rows vary but cols is fixed at 512, we can use a fixed block size
    # or a block size that divides 512. 512 is a multiple of 128, 256, 512.
    vals = tl.load(x_ptr + row_idx * cols + offs, mask=mask, other=-float('inf')))
    
    # Reduction
    max_val = tl.max(vals, axis=0)
    
    # Store the result
    tl.store(out_ptr + row_idx, max_val[0], mask=None)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape is (rows, 512)
        rows = x.shape[0]
        out = torch.empty((rows,), device=x.device, dtype=x.dtype)
        
        # Since the dimension is fixed at 512, we can use a block size of 512
        # or any power of 2 that divides 512.
        BLOCK = 512
        
        grid = lambda meta: (rows,) if rows > 0 else (1,))
        
        max_kernel[grid](
            x, 
            out, 
            num_rows=rows, 
            BLOCK=BLOCK
        )
        
        return out