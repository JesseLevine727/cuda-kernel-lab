# Task Suite Report

Suite: `reduced_v0`

Small local KernelBench-style suite used for native CUDA/Triton smoke baselines. Do not train on heldout_eval tasks.

- Dev: `['affine_1d', 'leaky_relu_1d']`
- Train trace: `[]`
- Heldout eval: `['fused_square_relu_1d', 'row_mean_2d', 'row_max_2d', 'row_softmax_2d', 'layer_norm_2d', 'matmul_2d']`
- All tasks: `['affine_1d', 'leaky_relu_1d', 'fused_square_relu_1d', 'row_mean_2d', 'row_max_2d', 'row_softmax_2d', 'layer_norm_2d', 'matmul_2d']`

## Tasks

### affine_1d

- Name: 1D affine transform
- Level: 1
- Supported backends: `('cuda_cpp', 'triton')`
- Reference:
```python
def forward(self, x: torch.Tensor, scale: float, bias: float) -> torch.Tensor:
    return x * scale + bias
```

### leaky_relu_1d

- Name: 1D LeakyReLU
- Level: 1
- Supported backends: `('cuda_cpp', 'triton')`
- Reference:
```python
def forward(self, x: torch.Tensor, negative_slope: float) -> torch.Tensor:
    return torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)
```

### fused_square_relu_1d

- Name: 1D fused square plus ReLU
- Level: 2
- Supported backends: `('cuda_cpp', 'triton')`
- Reference:
```python
def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return torch.relu(x * x + z)
```

### row_mean_2d

- Name: 2D row mean reduction
- Level: 1
- Supported backends: `('cuda_cpp', 'triton')`
- Reference:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x, dim=1)
```

### row_max_2d

- Name: 2D row max reduction
- Level: 1
- Supported backends: `('cuda_cpp', 'triton')`
- Reference:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.max(x, dim=1).values
```

### row_softmax_2d

- Name: 2D row softmax
- Level: 1
- Supported backends: `('cuda_cpp', 'triton')`
- Reference:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(x, dim=1)
```

### layer_norm_2d

- Name: 2D layer norm over columns
- Level: 1
- Supported backends: `('cuda_cpp', 'triton')`
- Reference:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.layer_norm(x, (x.shape[-1],))
```

### matmul_2d

- Name: Small matrix multiplication
- Level: 1
- Supported backends: `('cuda_cpp', 'triton')`
- Reference:
```python
def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b
```
