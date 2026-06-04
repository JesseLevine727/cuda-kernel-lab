#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor fused_mul_add(torch::Tensor x, torch::Tensor y, torch::Tensor z, float alpha);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fused_mul_add", &fused_mul_add); }
