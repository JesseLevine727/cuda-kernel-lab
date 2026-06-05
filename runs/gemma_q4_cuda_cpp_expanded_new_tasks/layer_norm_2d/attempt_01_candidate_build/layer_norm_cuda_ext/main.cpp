#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor layer_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("layer_norm_cuda", &layer_norm_cuda); }
