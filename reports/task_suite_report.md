# Task Suite Report

Suite: `reduced_v0`

Small local KernelBench-style suite used for native CUDA/Triton smoke baselines and trace collection. Do not train on `heldout_eval` tasks.

- Dev: `['affine_1d', 'leaky_relu_1d']`
- Train trace: 24 tasks
- Heldout eval: `['fused_square_relu_1d', 'row_mean_2d', 'row_max_2d', 'row_softmax_2d', 'layer_norm_2d', 'matmul_2d']`
- All tasks: 32

## Original Eval Tasks

| Task | Role | Reference |
| --- | --- | --- |
| `affine_1d` | dev | `x * scale + bias` |
| `leaky_relu_1d` | dev | `F.leaky_relu(x, negative_slope)` |
| `fused_square_relu_1d` | heldout | `relu(x * x + z)` |
| `row_mean_2d` | heldout | `mean(x, dim=1)` |
| `row_max_2d` | heldout | `max(x, dim=1).values` |
| `row_softmax_2d` | heldout | `softmax(x, dim=1)` |
| `layer_norm_2d` | heldout | `layer_norm(x, (x.shape[-1],))` |
| `matmul_2d` | heldout | `a @ b` |

## Train-Trace Tasks

| Batch | Tasks |
| --- | --- |
| v1 | `vector_add_1d`, `vector_mul_1d`, `axpy_1d`, `sigmoid_1d`, `clamp_1d`, `row_sum_2d`, `row_min_2d`, `column_scale_2d` |
| v2 | `scalar_add_1d`, `negate_1d`, `square_1d`, `abs_1d`, `relu_bias_1d`, `binary_sub_1d`, `column_bias_2d`, `row_mean_256_2d` |
| v3 | `scalar_mul_1d`, `weighted_sum_1d`, `addcmul_1d`, `relu_mul_1d`, `threshold_1d`, `tanh_1d`, `column_mul_add_2d`, `column_relu_bias_2d` |

All tasks support both `cuda_cpp` and `triton`, but current trace collection targets `cuda_cpp`.
