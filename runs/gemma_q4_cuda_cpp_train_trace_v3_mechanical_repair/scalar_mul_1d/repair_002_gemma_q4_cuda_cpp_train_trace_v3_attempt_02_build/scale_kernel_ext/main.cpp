#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor scale(torch::Tensor x, float scale);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("scale", &scale); }
