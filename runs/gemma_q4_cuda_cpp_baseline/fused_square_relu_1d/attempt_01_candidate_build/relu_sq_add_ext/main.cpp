#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor relu_sq_add(torch::Tensor x, torch::Tensor z);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("relu_sq_add", &relu_sq_add); }
