#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor relu_mul(torch::Tensor x, torch::Tensor y);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("relu_mul", &relu_mul); }
