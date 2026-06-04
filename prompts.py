from __future__ import annotations

from cuda_kernel_lab.tasks import KernelTask


ONE_SHOT_TRITON = r'''
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def add_kernel(a, b, out, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    av = tl.load(a + offs, mask=mask, other=0.0)
    bv = tl.load(b + offs, mask=mask, other=0.0)
    tl.store(out + offs, av + bv, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        a = a.contiguous()
        b = b.contiguous()
        out = torch.empty_like(a)
        n = a.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
        add_kernel[grid](a, b, out, n, BLOCK=256)
        return out
'''.strip()

ONE_SHOT_CUDA_CPP = r'''
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


CUDA_SOURCE = r"""
#include <torch/extension.h>

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

torch::Tensor add(torch::Tensor a, torch::Tensor b) {
  auto out = torch::empty_like(a);
  int n = a.numel();
  add_kernel<<<(n + 255) / 256, 256>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
}
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor add(torch::Tensor a, torch::Tensor b);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("add", &add); }
"""

ext = load_inline(
    name="example_add_ext",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=None,
    extra_cuda_cflags=["-O2"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return ext.add(a.contiguous(), b.contiguous())
'''.strip()


SYSTEM_PROMPT = """You write small, correct GPU kernels for benchmark evaluation.
Return only Python code.
The code must define class ModelNew(torch.nn.Module).
Do not monkey-patch torch, do not read or write files, do not use subprocesses, and do not print.
Prioritize correctness over speed.
"""


def build_prompt(task: KernelTask, feedback: str | None = None, backend: str = "triton") -> str:
    if backend == "cuda_cpp":
        example = ONE_SHOT_CUDA_CPP
        backend_requirements = [
            "- Use native CUDA C++ with torch.utils.cpp_extension.load_inline.",
            "- Include CUDA_SOURCE and CPP_SOURCE strings inside the Python module.",
            "- Set functions=None and define PYBIND11_MODULE in CPP_SOURCE.",
            "- Use verbose=False in load_inline.",
            "- Do not call subprocess or read/write files.",
            "- The evaluator provides CUDA_HOME and TORCH_CUDA_ARCH_LIST=12.0.",
        ]
    else:
        example = ONE_SHOT_TRITON
        backend_requirements = [
            "- Use @triton.jit kernels where appropriate.",
            "- Return only Python code, preferably in one ```python fenced block.",
        ]
    parts = [
        "Write a complete Python module for this KernelBench-style task.",
        f"Backend: {backend}",
        "",
        "Example of the expected style:",
        "```python",
        example,
        "```",
        "",
        "Task:",
        task.prompt,
        "",
        "PyTorch reference forward:",
        "```python",
        task.reference_code,
        "```",
        "",
        "Requirements:",
        "- Define exactly one class named ModelNew that subclasses torch.nn.Module.",
        "- The class constructor must take no required arguments.",
        "- Match the forward signature described in the task.",
        "- Inputs are already CUDA float32 tensors unless otherwise noted.",
        "- Use torch.empty_like or torch.empty for outputs.",
        *backend_requirements,
    ]
    if feedback:
        parts.extend(["", "Previous attempt failed. Fix this issue:", feedback[-4000:]])
    return "\n".join(parts)
