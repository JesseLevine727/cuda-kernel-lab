#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor linear(torch::Tensor x, torch::Tensor weights, torch::Tensor bias);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("linear", &linear); }
