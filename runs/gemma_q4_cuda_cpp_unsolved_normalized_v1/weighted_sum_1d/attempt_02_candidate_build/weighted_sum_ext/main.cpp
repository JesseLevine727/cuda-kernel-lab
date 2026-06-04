#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor weighted_sum(torch::Tensor x, torch::Tensor y, float alpha, float beta);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("weighted_sum", &weighted_sum); }
