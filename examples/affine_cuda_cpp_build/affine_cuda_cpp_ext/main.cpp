#include <torch/extension.h>

#include <torch/extension.h>

torch::Tensor affine(torch::Tensor x, double scale, double bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("affine", &affine);
}
