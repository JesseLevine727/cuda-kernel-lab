#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor linear(torch::Tensor x, float scale, float bias);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("linear", &linear); }
