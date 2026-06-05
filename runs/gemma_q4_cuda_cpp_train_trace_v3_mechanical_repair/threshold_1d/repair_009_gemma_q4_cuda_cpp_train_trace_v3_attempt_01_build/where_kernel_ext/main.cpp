#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor where_func(torch::Tensor x, float threshold, float value);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("where_func", &where_func); }
