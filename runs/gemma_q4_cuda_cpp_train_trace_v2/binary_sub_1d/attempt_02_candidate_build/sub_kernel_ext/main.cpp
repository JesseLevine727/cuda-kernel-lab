#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor sub(torch::Tensor x, torch::Tensor z);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("sub", &sub); }
