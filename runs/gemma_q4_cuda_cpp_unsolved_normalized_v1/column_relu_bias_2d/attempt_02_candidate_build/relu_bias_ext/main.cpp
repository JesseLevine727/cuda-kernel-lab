#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor relu_bias(torch::Tensor x, torch::Tensor bias);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("relu_bias", &relu_bias); }
