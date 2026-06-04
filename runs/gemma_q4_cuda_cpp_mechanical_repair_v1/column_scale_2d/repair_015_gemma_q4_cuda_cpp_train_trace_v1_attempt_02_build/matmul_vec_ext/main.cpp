#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor matmul_vec(torch::Tensor x, torch::Tensor weights);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("matmul_vec", &matmul_vec); }
