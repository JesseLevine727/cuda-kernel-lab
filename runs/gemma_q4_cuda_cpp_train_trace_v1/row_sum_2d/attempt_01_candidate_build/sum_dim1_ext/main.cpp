#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor sum_dim1(torch::Tensor x);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sum_dim1", &sum_dim1);
}
