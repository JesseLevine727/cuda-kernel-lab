#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor min_dim1(torch::Tensor input);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   m.def("min_dim1", &min_dim1);
 }
