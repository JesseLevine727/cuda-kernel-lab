#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor square(torch::Tensor x);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("square", &square); }
