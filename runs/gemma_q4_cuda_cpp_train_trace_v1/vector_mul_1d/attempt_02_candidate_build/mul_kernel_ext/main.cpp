#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor mul(torch::Tensor a, torch::Tensor b);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("mul", &mul); }
