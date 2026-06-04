#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor matmul(torchെയTensor a, torch::Tensor b);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("matmul", &matmul); }
