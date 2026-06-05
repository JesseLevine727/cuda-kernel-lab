#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor matmul_vec_op(torch::Tensor x, torch::Tensor weights);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("matmul_vec_op", &matmul_vec_op); }
