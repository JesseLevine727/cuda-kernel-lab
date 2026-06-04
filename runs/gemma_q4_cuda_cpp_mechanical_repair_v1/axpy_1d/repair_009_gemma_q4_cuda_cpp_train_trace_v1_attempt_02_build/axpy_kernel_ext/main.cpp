#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor axpy(torch::Tensor x, torch::Tensor y, float alpha);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("axpy", &axpy); }
