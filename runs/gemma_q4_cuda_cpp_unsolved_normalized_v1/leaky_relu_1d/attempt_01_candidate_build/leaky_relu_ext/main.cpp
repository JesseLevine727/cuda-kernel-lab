#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor leaky_relu(torch::Tensor x, float negative_slope);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("leaky_relu", &leaky_relu); }
