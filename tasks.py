from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


InputFactory = Callable[[int, torch.device], tuple]
ReferenceFn = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class KernelTask:
    task_id: str
    name: str
    level: int
    backend: str
    supported_backends: tuple[str, ...]
    prompt: str
    reference_code: str
    input_factory: InputFactory
    reference: ReferenceFn
    atol: float = 1e-4
    rtol: float = 1e-4


def _randn(shape: tuple[int, ...], seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randn(shape, generator=gen, dtype=torch.float32).to(device)


def _rand(shape: tuple[int, ...], seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.rand(shape, generator=gen, dtype=torch.float32).to(device)


def affine_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor, float, float]:
    sizes = (65_536, 131_072, 262_144)
    return (_randn((sizes[seed % len(sizes)],), seed, device), 1.75, -0.125)


def affine_reference(x: torch.Tensor, scale: float, bias: float) -> torch.Tensor:
    return x * scale + bias


def leaky_relu_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor, float]:
    sizes = (65_536, 131_072, 262_144)
    return (_randn((sizes[seed % len(sizes)],), seed, device), 0.1)


def leaky_relu_reference(x: torch.Tensor, negative_slope: float) -> torch.Tensor:
    return F.leaky_relu(x, negative_slope=negative_slope)


def fused_square_relu_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sizes = (65_536, 131_072, 262_144)
    shape = (sizes[seed % len(sizes)],)
    return (_randn(shape, seed, device), _randn(shape, seed + 1, device))


def fused_square_relu_reference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.relu(x * x + y)


def row_mean_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor]:
    rows = (256, 512, 1024)
    return (_rand((rows[seed % len(rows)], 512), seed, device),)


def row_mean_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x, dim=1)


def row_max_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor]:
    rows = (256, 512, 1024)
    return (_rand((rows[seed % len(rows)], 512), seed, device),)


def row_max_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.max(x, dim=1).values


def row_softmax_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor]:
    rows = (128, 256, 512)
    return (_randn((rows[seed % len(rows)], 256), seed, device),)


def row_softmax_reference(x: torch.Tensor) -> torch.Tensor:
    return F.softmax(x, dim=1)


def layer_norm_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor]:
    rows = (128, 256, 512)
    return (_randn((rows[seed % len(rows)], 256), seed, device),)


def layer_norm_reference(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],))


def matmul_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    shapes = ((128, 128, 128), (256, 128, 256), (64, 256, 128))
    m, k, n = shapes[seed % len(shapes)]
    return (_randn((m, k), seed, device), _randn((k, n), seed + 1, device))


def matmul_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


TASKS: dict[str, KernelTask] = {
    "affine_1d": KernelTask(
        task_id="affine_1d",
        name="1D affine transform",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement y = x * scale + bias for a contiguous float32 CUDA tensor x. "
            "The forward signature must be forward(self, x, scale: float, bias: float). "
            "Return a tensor with the same shape as x. x is 1D and its length may vary."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor, scale: float, bias: float) -> torch.Tensor:\n"
            "    return x * scale + bias"
        ),
        input_factory=affine_inputs,
        reference=affine_reference,
    ),
    "leaky_relu_1d": KernelTask(
        task_id="leaky_relu_1d",
        name="1D LeakyReLU",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement torch.nn.functional.leaky_relu for a contiguous float32 CUDA tensor x. "
            "The forward signature must be forward(self, x, negative_slope: float). "
            "Return a tensor with the same shape as x. x is 1D and its length may vary."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor, negative_slope: float) -> torch.Tensor:\n"
            "    return torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)"
        ),
        input_factory=leaky_relu_inputs,
        reference=leaky_relu_reference,
    ),
    "fused_square_relu_1d": KernelTask(
        task_id="fused_square_relu_1d",
        name="1D fused square plus ReLU",
        level=2,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement y = relu(x * x + z) for contiguous float32 CUDA tensors x and z. "
            "The forward signature must be forward(self, x, z). "
            "Return a tensor with the same shape as x. x and z are 1D and their length may vary."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:\n"
            "    return torch.relu(x * x + z)"
        ),
        input_factory=fused_square_relu_inputs,
        reference=fused_square_relu_reference,
    ),
    "row_mean_2d": KernelTask(
        task_id="row_mean_2d",
        name="2D row mean reduction",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement torch.mean(x, dim=1) for a contiguous float32 CUDA tensor x with shape (rows, 512). "
            "The forward signature must be forward(self, x). rows may vary. Return a 1D tensor of length rows."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
            "    return torch.mean(x, dim=1)"
        ),
        input_factory=row_mean_inputs,
        reference=row_mean_reference,
        atol=2e-4,
        rtol=2e-4,
    ),
    "row_max_2d": KernelTask(
        task_id="row_max_2d",
        name="2D row max reduction",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement torch.max(x, dim=1).values for a contiguous float32 CUDA tensor x with shape (rows, 512). "
            "The forward signature must be forward(self, x). rows may vary. Return a 1D tensor of length rows."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
            "    return torch.max(x, dim=1).values"
        ),
        input_factory=row_max_inputs,
        reference=row_max_reference,
    ),
    "row_softmax_2d": KernelTask(
        task_id="row_softmax_2d",
        name="2D row softmax",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement torch.nn.functional.softmax(x, dim=1) for a contiguous float32 CUDA tensor x with shape (rows, 256). "
            "The forward signature must be forward(self, x). rows may vary. Return a tensor with the same shape as x."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
            "    return torch.nn.functional.softmax(x, dim=1)"
        ),
        input_factory=row_softmax_inputs,
        reference=row_softmax_reference,
        atol=2e-4,
        rtol=2e-4,
    ),
    "layer_norm_2d": KernelTask(
        task_id="layer_norm_2d",
        name="2D layer norm over columns",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement torch.nn.functional.layer_norm(x, (x.shape[-1],)) for a contiguous float32 CUDA tensor x with shape (rows, 256). "
            "The forward signature must be forward(self, x). rows may vary. Return a tensor with the same shape as x."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
            "    return torch.nn.functional.layer_norm(x, (x.shape[-1],))"
        ),
        input_factory=layer_norm_inputs,
        reference=layer_norm_reference,
        atol=3e-4,
        rtol=3e-4,
    ),
    "matmul_2d": KernelTask(
        task_id="matmul_2d",
        name="Small matrix multiplication",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement matrix multiplication a @ b for contiguous float32 CUDA tensors. "
            "The forward signature must be forward(self, a, b). Shapes may vary among small 2D matrices."
        ),
        reference_code=(
            "def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:\n"
            "    return a @ b"
        ),
        input_factory=matmul_inputs,
        reference=matmul_reference,
        atol=3e-4,
        rtol=3e-4,
    ),
}


def get_task(task_id: str) -> KernelTask:
    try:
        return TASKS[task_id]
    except KeyError as exc:
        known = ", ".join(sorted(TASKS))
        raise KeyError(f"Unknown task {task_id!r}. Known tasks: {known}") from exc
