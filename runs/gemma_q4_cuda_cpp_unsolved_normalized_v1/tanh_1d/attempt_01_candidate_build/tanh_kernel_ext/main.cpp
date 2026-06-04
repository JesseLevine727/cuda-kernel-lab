#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor tanh_func(torch::Tensor x);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("tanh_func", &tanh_func); }
