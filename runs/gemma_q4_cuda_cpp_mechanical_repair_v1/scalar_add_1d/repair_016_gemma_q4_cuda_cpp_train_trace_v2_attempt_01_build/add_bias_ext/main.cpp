#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor add_bias(torch::Tensor x, float bias);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("add_bias", &add_bias); }
