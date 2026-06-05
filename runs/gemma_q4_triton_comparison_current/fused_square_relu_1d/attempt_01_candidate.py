import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_sq_add_kernel(x, z, out, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    
    x_val = tl.load(x + offs, mask=mask, other=0.0)
    z_val = tl.load(z + offs, mask=mask, other=0.0)
    
    res = x_val * x_val + z_val
    res = tl.maximum(res, 0.0)
    
    tl.store(out + offs, res, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        z = z.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
        
        relu_sq_add_kernel[grid](
            x, 
            z, 
            out, 
            n, 
            BLOCK=256
        )
        return out