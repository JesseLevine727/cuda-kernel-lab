#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor mean_kernel_func(torch::Tensor input);
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("mean_kernel_func", &mean_kernel_func); }
