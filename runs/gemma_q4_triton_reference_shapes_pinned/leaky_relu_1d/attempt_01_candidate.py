import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def leaky_relu_kernel(x, out, negative_slope: tl.🙏, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    n = x.shape[0]
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    x_val = tl.load(x + offsets, mask=mask, other=0.0)
    
    # leaky_relu: x if x > 0 else negative_slope * x
    res = tl.where(x_val > 0.0, x_val, x_val * negative_slope)
    
    tl.store(out + offsets, res, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, negative_slope: float) -> torch.Tensor:
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        
        # Define the grid based on the block size
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
        
        leaky_relu_kernel[grid](
            x, 
            out, 
            negative_slope=float(negative_slope)),
            BLOCK=BLOCK_SIZE
        )
        return out