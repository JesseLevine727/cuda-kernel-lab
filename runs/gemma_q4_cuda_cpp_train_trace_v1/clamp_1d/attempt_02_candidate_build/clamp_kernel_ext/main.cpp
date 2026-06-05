#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor clamp(torch::Tensor x, float min_val, float max_val);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("clamp", &clamp); }
