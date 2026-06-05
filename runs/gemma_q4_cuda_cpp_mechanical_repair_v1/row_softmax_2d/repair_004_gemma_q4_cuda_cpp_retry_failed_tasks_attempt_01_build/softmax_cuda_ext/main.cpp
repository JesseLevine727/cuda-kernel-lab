#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor softmax_cuda(torch::Tensor x);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("softmax_cuda", &softmax_cuda); }
