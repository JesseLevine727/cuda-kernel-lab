#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor tanh(torch::Tensor x);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("tanh", &tanh); }
