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
    
    # Ensure scale and bias are treated as scalars in the expression
    # The error in the previous attempt was likely due to type ambiguity 
    # in the compiler's IR generation for the arithmetic operation.
    res = x_val * scale + bias
    
    tl.store(out + offs, res, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, scale: float, bias: float) -> torch.Tensor:
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        
        # Define the grid based on the block size
        BLOCK = 128
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
        
        # Launch the kernel
        linear_kernel[grid](
            x, 
            torch.tensor(scale, device=x.device, dtype=torch.float32), 
            torch.tensor(bias, device=x.device, dtype=torch.float32), 
            out, 
            n, 
            BLOCK=BLOCK
        )
        return out