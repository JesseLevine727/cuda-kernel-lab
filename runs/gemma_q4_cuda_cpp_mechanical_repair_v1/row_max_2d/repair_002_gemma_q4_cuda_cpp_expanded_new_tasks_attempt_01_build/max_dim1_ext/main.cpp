#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor max_dim1(torch::Tensor input);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("max_dim1", &max_dim1); }
