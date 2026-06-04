import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def affine_kernel(x, out, scale: tl.constexpr, bias: tl.constexpr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    xv = tl.load(x + offs, mask=mask, other=0.0)
    tl.store(out + offs, xv * scale + bias, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, scale: float, bias: float):
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
        affine_kernel[grid](x, out, scale, bias, n, BLOCK=256)
        return out

