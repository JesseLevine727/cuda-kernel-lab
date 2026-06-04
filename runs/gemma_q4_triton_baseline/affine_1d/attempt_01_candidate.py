import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def linear_kernel(x, scale, bias, out, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    
    x_val = tl.load(x + offs, mask=mask, other=0.0)
    
    # y = x * scale + bias
    res = x_val * scale + bias
    
    tl.store(out + offs, res, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, scale: float, bias: float) -> torch.Tensor:
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        
        # Use a power of 2 for BLOCK size for simplicity
        BLOCK_SIZE = 1024
        
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
        
        linear_kernel[grid](
            x, 
            torch.tensor(scale, device=x.device), 
            torch.tensor(bias, device=x.device), 
            out, 
            n, 
            BLOCK=BLOCK_SIZE
        )
        
        return out