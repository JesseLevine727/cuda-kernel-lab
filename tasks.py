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


def vector_add_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sizes = (65_536, 131_072, 262_144)
    shape = (sizes[seed % len(sizes)],)
    return (_randn(shape, seed, device), _randn(shape, seed + 1, device))


def vector_add_reference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


def vector_mul_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sizes = (65_536, 131_072, 262_144)
    shape = (sizes[seed % len(sizes)],)
    return (_randn(shape, seed, device), _randn(shape, seed + 1, device))


def vector_mul_reference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x * y


def axpy_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    sizes = (65_536, 131_072, 262_144)
    shape = (sizes[seed % len(sizes)],)
    return (_randn(shape, seed, device), _randn(shape, seed + 1, device), 0.25)


def axpy_reference(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    return alpha * x + y


def sigmoid_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor]:
    sizes = (65_536, 131_072, 262_144)
    return (_randn((sizes[seed % len(sizes)],), seed, device),)


def sigmoid_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def clamp_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor, float, float]:
    sizes = (65_536, 131_072, 262_144)
    return (_randn((sizes[seed % len(sizes)],), seed, device), -0.5, 0.5)


def clamp_reference(x: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    return torch.clamp(x, min=min_value, max=max_value)


def row_sum_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor]:
    rows = (256, 512, 1024)
    return (_randn((rows[seed % len(rows)], 256), seed, device),)


def row_sum_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x, dim=1)


def row_min_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor]:
    rows = (256, 512, 1024)
    return (_rand((rows[seed % len(rows)], 256), seed, device),)


def row_min_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.min(x, dim=1).values


def column_scale_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rows = (128, 256, 512)
    return (
        _randn((rows[seed % len(rows)], 256), seed, device),
        _randn((256,), seed + 1, device),
    )


def column_scale_reference(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return x * weights


def scalar_add_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor, float]:
    sizes = (65_536, 131_072, 262_144)
    return (_randn((sizes[seed % len(sizes)],), seed, device), 0.375)


def scalar_add_reference(x: torch.Tensor, bias: float) -> torch.Tensor:
    return x + bias


def negate_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor]:
    sizes = (65_536, 131_072, 262_144)
    return (_randn((sizes[seed % len(sizes)],), seed, device),)


def negate_reference(x: torch.Tensor) -> torch.Tensor:
    return -x


def square_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor]:
    sizes = (65_536, 131_072, 262_144)
    return (_randn((sizes[seed % len(sizes)],), seed, device),)


def square_reference(x: torch.Tensor) -> torch.Tensor:
    return x * x


def abs_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor]:
    sizes = (65_536, 131_072, 262_144)
    return (_randn((sizes[seed % len(sizes)],), seed, device),)


def abs_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x)


def relu_bias_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor, float]:
    sizes = (65_536, 131_072, 262_144)
    return (_randn((sizes[seed % len(sizes)],), seed, device), -0.2)


def relu_bias_reference(x: torch.Tensor, bias: float) -> torch.Tensor:
    return torch.relu(x + bias)


def binary_sub_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sizes = (65_536, 131_072, 262_144)
    shape = (sizes[seed % len(sizes)],)
    return (_randn(shape, seed, device), _randn(shape, seed + 1, device))


def binary_sub_reference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x - y


def column_bias_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rows = (128, 256, 512)
    return (
        _randn((rows[seed % len(rows)], 256), seed, device),
        _randn((256,), seed + 1, device),
    )


def column_bias_reference(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x + bias


def row_mean_256_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor]:
    rows = (256, 512, 1024)
    return (_randn((rows[seed % len(rows)], 256), seed, device),)


def row_mean_256_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x, dim=1)


def scalar_mul_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor, float]:
    sizes = (65_536, 131_072, 262_144)
    return (_randn((sizes[seed % len(sizes)],), seed, device), -1.25)


def scalar_mul_reference(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x * scale


def weighted_sum_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    sizes = (65_536, 131_072, 262_144)
    shape = (sizes[seed % len(sizes)],)
    return (_randn(shape, seed, device), _randn(shape, seed + 1, device), 0.75, -0.25)


def weighted_sum_reference(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    return alpha * x + beta * y


def addcmul_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    sizes = (65_536, 131_072, 262_144)
    shape = (sizes[seed % len(sizes)],)
    return (
        _randn(shape, seed, device),
        _randn(shape, seed + 1, device),
        _randn(shape, seed + 2, device),
        0.5,
    )


def addcmul_reference(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    return x + alpha * y * z


def relu_mul_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sizes = (65_536, 131_072, 262_144)
    shape = (sizes[seed % len(sizes)],)
    return (_randn(shape, seed, device), _randn(shape, seed + 1, device))


def relu_mul_reference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.relu(x) * y


def threshold_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor, float, float]:
    sizes = (65_536, 131_072, 262_144)
    return (_randn((sizes[seed % len(sizes)],), seed, device), 0.2, -0.75)


def threshold_reference(x: torch.Tensor, threshold: float, value: float) -> torch.Tensor:
    return torch.where(x > threshold, x, torch.full_like(x, value))


def tanh_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor]:
    sizes = (65_536, 131_072, 262_144)
    return (_randn((sizes[seed % len(sizes)],), seed, device),)


def tanh_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def column_mul_add_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = (128, 256, 512)
    return (
        _randn((rows[seed % len(rows)], 256), seed, device),
        _randn((256,), seed + 1, device),
        _randn((256,), seed + 2, device),
    )


def column_mul_add_reference(
    x: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return x * weights + bias


def column_relu_bias_inputs(seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rows = (128, 256, 512)
    return (
        _randn((rows[seed % len(rows)], 256), seed, device),
        _randn((256,), seed + 1, device),
    )


def column_relu_bias_reference(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.relu(x + bias)


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
    "vector_add_1d": KernelTask(
        task_id="vector_add_1d",
        name="1D vector add",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement y = x + z for contiguous float32 CUDA tensors x and z. "
            "The forward signature must be forward(self, x, z). "
            "Return a tensor with the same shape as x. x and z are 1D and their length may vary."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:\n"
            "    return x + z"
        ),
        input_factory=vector_add_inputs,
        reference=vector_add_reference,
    ),
    "vector_mul_1d": KernelTask(
        task_id="vector_mul_1d",
        name="1D vector multiply",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement y = x * z for contiguous float32 CUDA tensors x and z. "
            "The forward signature must be forward(self, x, z). "
            "Return a tensor with the same shape as x. x and z are 1D and their length may vary."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:\n"
            "    return x * z"
        ),
        input_factory=vector_mul_inputs,
        reference=vector_mul_reference,
    ),
    "axpy_1d": KernelTask(
        task_id="axpy_1d",
        name="1D AXPY fused op",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement out = alpha * x + y for contiguous float32 CUDA tensors x and y. "
            "The forward signature must be forward(self, x, y, alpha: float). "
            "Return a tensor with the same shape as x. x and y are 1D and their length may vary."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:\n"
            "    return alpha * x + y"
        ),
        input_factory=axpy_inputs,
        reference=axpy_reference,
    ),
    "sigmoid_1d": KernelTask(
        task_id="sigmoid_1d",
        name="1D sigmoid",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement torch.sigmoid(x) for a contiguous float32 CUDA tensor x. "
            "The forward signature must be forward(self, x). "
            "Return a tensor with the same shape as x. x is 1D and its length may vary."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
            "    return torch.sigmoid(x)"
        ),
        input_factory=sigmoid_inputs,
        reference=sigmoid_reference,
        atol=2e-4,
        rtol=2e-4,
    ),
    "clamp_1d": KernelTask(
        task_id="clamp_1d",
        name="1D clamp",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement torch.clamp(x, min=min_value, max=max_value) for a contiguous float32 CUDA tensor x. "
            "The forward signature must be forward(self, x, min_value: float, max_value: float). "
            "Return a tensor with the same shape as x. x is 1D and its length may vary."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:\n"
            "    return torch.clamp(x, min=min_value, max=max_value)"
        ),
        input_factory=clamp_inputs,
        reference=clamp_reference,
    ),
    "row_sum_2d": KernelTask(
        task_id="row_sum_2d",
        name="2D row sum reduction",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement torch.sum(x, dim=1) for a contiguous float32 CUDA tensor x with shape (rows, 256). "
            "The forward signature must be forward(self, x). rows may vary. Return a 1D tensor of length rows."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
            "    return torch.sum(x, dim=1)"
        ),
        input_factory=row_sum_inputs,
        reference=row_sum_reference,
        atol=4e-4,
        rtol=4e-4,
    ),
    "row_min_2d": KernelTask(
        task_id="row_min_2d",
        name="2D row min reduction",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement torch.min(x, dim=1).values for a contiguous float32 CUDA tensor x with shape (rows, 256). "
            "The forward signature must be forward(self, x). rows may vary. Return a 1D tensor of length rows."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
            "    return torch.min(x, dim=1).values"
        ),
        input_factory=row_min_inputs,
        reference=row_min_reference,
    ),
    "column_scale_2d": KernelTask(
        task_id="column_scale_2d",
        name="2D column scale",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement out = x * weights for a contiguous float32 CUDA tensor x with shape (rows, 256) "
            "and a contiguous float32 CUDA tensor weights with shape (256,). "
            "The forward signature must be forward(self, x, weights). rows may vary. Return a tensor with the same shape as x."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:\n"
            "    return x * weights"
        ),
        input_factory=column_scale_inputs,
        reference=column_scale_reference,
    ),
    "scalar_add_1d": KernelTask(
        task_id="scalar_add_1d",
        name="1D scalar add",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement y = x + bias for a contiguous float32 CUDA tensor x. "
            "The forward signature must be forward(self, x, bias: float). "
            "Return a tensor with the same shape as x. x is 1D and its length may vary."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor, bias: float) -> torch.Tensor:\n"
            "    return x + bias"
        ),
        input_factory=scalar_add_inputs,
        reference=scalar_add_reference,
    ),
    "negate_1d": KernelTask(
        task_id="negate_1d",
        name="1D negate",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement y = -x for a contiguous float32 CUDA tensor x. "
            "The forward signature must be forward(self, x). "
            "Return a tensor with the same shape as x. x is 1D and its length may vary."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
            "    return -x"
        ),
        input_factory=negate_inputs,
        reference=negate_reference,
    ),
    "square_1d": KernelTask(
        task_id="square_1d",
        name="1D square",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement y = x * x for a contiguous float32 CUDA tensor x. "
            "The forward signature must be forward(self, x). "
            "Return a tensor with the same shape as x. x is 1D and its length may vary."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
            "    return x * x"
        ),
        input_factory=square_inputs,
        reference=square_reference,
    ),
    "abs_1d": KernelTask(
        task_id="abs_1d",
        name="1D absolute value",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement torch.abs(x) for a contiguous float32 CUDA tensor x. "
            "The forward signature must be forward(self, x). "
            "Return a tensor with the same shape as x. x is 1D and its length may vary."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
            "    return torch.abs(x)"
        ),
        input_factory=abs_inputs,
        reference=abs_reference,
    ),
    "relu_bias_1d": KernelTask(
        task_id="relu_bias_1d",
        name="1D ReLU plus bias",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement y = relu(x + bias) for a contiguous float32 CUDA tensor x. "
            "The forward signature must be forward(self, x, bias: float). "
            "Return a tensor with the same shape as x. x is 1D and its length may vary."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor, bias: float) -> torch.Tensor:\n"
            "    return torch.relu(x + bias)"
        ),
        input_factory=relu_bias_inputs,
        reference=relu_bias_reference,
    ),
    "binary_sub_1d": KernelTask(
        task_id="binary_sub_1d",
        name="1D vector subtract",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement y = x - z for contiguous float32 CUDA tensors x and z. "
            "The forward signature must be forward(self, x, z). "
            "Return a tensor with the same shape as x. x and z are 1D and their length may vary."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:\n"
            "    return x - z"
        ),
        input_factory=binary_sub_inputs,
        reference=binary_sub_reference,
    ),
    "column_bias_2d": KernelTask(
        task_id="column_bias_2d",
        name="2D column bias",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement out = x + bias for a contiguous float32 CUDA tensor x with shape (rows, 256) "
            "and a contiguous float32 CUDA tensor bias with shape (256,). "
            "The forward signature must be forward(self, x, bias). rows may vary. Return a tensor with the same shape as x."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:\n"
            "    return x + bias"
        ),
        input_factory=column_bias_inputs,
        reference=column_bias_reference,
    ),
    "row_mean_256_2d": KernelTask(
        task_id="row_mean_256_2d",
        name="2D row mean reduction width 256",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement torch.mean(x, dim=1) for a contiguous float32 CUDA tensor x with shape (rows, 256). "
            "The forward signature must be forward(self, x). rows may vary. Return a 1D tensor of length rows."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
            "    return torch.mean(x, dim=1)"
        ),
        input_factory=row_mean_256_inputs,
        reference=row_mean_256_reference,
        atol=4e-4,
        rtol=4e-4,
    ),
    "scalar_mul_1d": KernelTask(
        task_id="scalar_mul_1d",
        name="1D scalar multiply",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement y = x * scale for a contiguous float32 CUDA tensor x. "
            "The forward signature must be forward(self, x, scale: float). "
            "Return a tensor with the same shape as x. x is 1D and its length may vary."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor, scale: float) -> torch.Tensor:\n"
            "    return x * scale"
        ),
        input_factory=scalar_mul_inputs,
        reference=scalar_mul_reference,
    ),
    "weighted_sum_1d": KernelTask(
        task_id="weighted_sum_1d",
        name="1D weighted sum",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement out = alpha * x + beta * y for contiguous float32 CUDA tensors x and y. "
            "The forward signature must be forward(self, x, y, alpha: float, beta: float). "
            "Return a tensor with the same shape as x. x and y are 1D and their length may vary."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:\n"
            "    return alpha * x + beta * y"
        ),
        input_factory=weighted_sum_inputs,
        reference=weighted_sum_reference,
    ),
    "addcmul_1d": KernelTask(
        task_id="addcmul_1d",
        name="1D addcmul fused op",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement out = x + alpha * y * z for contiguous float32 CUDA tensors x, y, and z. "
            "The forward signature must be forward(self, x, y, z, alpha: float). "
            "Return a tensor with the same shape as x. x, y, and z are 1D and their length may vary."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, alpha: float) -> torch.Tensor:\n"
            "    return x + alpha * y * z"
        ),
        input_factory=addcmul_inputs,
        reference=addcmul_reference,
    ),
    "relu_mul_1d": KernelTask(
        task_id="relu_mul_1d",
        name="1D ReLU multiply",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement out = relu(x) * y for contiguous float32 CUDA tensors x and y. "
            "The forward signature must be forward(self, x, y). "
            "Return a tensor with the same shape as x. x and y are 1D and their length may vary."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:\n"
            "    return torch.relu(x) * y"
        ),
        input_factory=relu_mul_inputs,
        reference=relu_mul_reference,
    ),
    "threshold_1d": KernelTask(
        task_id="threshold_1d",
        name="1D threshold replacement",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement torch.where(x > threshold, x, value) for a contiguous float32 CUDA tensor x. "
            "The forward signature must be forward(self, x, threshold: float, value: float). "
            "Return a tensor with the same shape as x. x is 1D and its length may vary."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor, threshold: float, value: float) -> torch.Tensor:\n"
            "    return torch.where(x > threshold, x, torch.full_like(x, value))"
        ),
        input_factory=threshold_inputs,
        reference=threshold_reference,
    ),
    "tanh_1d": KernelTask(
        task_id="tanh_1d",
        name="1D tanh",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement torch.tanh(x) for a contiguous float32 CUDA tensor x. "
            "The forward signature must be forward(self, x). "
            "Return a tensor with the same shape as x. x is 1D and its length may vary."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
            "    return torch.tanh(x)"
        ),
        input_factory=tanh_inputs,
        reference=tanh_reference,
        atol=3e-4,
        rtol=3e-4,
    ),
    "column_mul_add_2d": KernelTask(
        task_id="column_mul_add_2d",
        name="2D column multiply plus bias",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement out = x * weights + bias for a contiguous float32 CUDA tensor x with shape (rows, 256), "
            "a contiguous float32 CUDA tensor weights with shape (256,), and a contiguous float32 CUDA tensor bias with shape (256,). "
            "The forward signature must be forward(self, x, weights, bias). rows may vary. Return a tensor with the same shape as x."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:\n"
            "    return x * weights + bias"
        ),
        input_factory=column_mul_add_inputs,
        reference=column_mul_add_reference,
    ),
    "column_relu_bias_2d": KernelTask(
        task_id="column_relu_bias_2d",
        name="2D column ReLU plus bias",
        level=1,
        backend="triton",
        supported_backends=("cuda_cpp", "triton"),
        prompt=(
            "Implement out = relu(x + bias) for a contiguous float32 CUDA tensor x with shape (rows, 256) "
            "and a contiguous float32 CUDA tensor bias with shape (256,). "
            "The forward signature must be forward(self, x, bias). rows may vary. Return a tensor with the same shape as x."
        ),
        reference_code=(
            "def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:\n"
            "    return torch.relu(x + bias)"
        ),
        input_factory=column_relu_bias_inputs,
        reference=column_relu_bias_reference,
    ),
}


def get_task(task_id: str) -> KernelTask:
    try:
        return TASKS[task_id]
    except KeyError as exc:
        known = ", ".join(sorted(TASKS))
        raise KeyError(f"Unknown task {task_id!r}. Known tasks: {known}") from exc
