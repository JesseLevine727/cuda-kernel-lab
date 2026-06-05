#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor negate(torch::Tensor input);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("negate", &negate); }
