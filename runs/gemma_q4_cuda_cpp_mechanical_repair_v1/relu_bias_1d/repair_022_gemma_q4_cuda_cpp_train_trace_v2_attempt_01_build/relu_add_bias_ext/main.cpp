#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor relu_add_bias(torch::Tensor x, float bias);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("relu_add_bias", &relu_add_bias); }
