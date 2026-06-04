#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor mean_dim1(torch::Tensor x);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("mean_dim1", &mean_dim1); }
