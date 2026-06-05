#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor mul(torch::Tensor x, torch::Tensor z);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("mul", &mul); }
