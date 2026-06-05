#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor abs_func(torch::Tensor x);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("abs_func", &abs_func); }
