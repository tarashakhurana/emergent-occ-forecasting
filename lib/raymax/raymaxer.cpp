#include <torch/extension.h>
#include <string>
#include <vector>

/*
 * CUDA forward declarations
 */
std::vector<torch::Tensor> argmax_cuda(
    torch::Tensor sigma,
    torch::Tensor origins,
    torch::Tensor points);

/*
 * C++ interface
 */
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
std::vector<torch::Tensor> argmax(
    torch::Tensor sigma,
    torch::Tensor origins,
    torch::Tensor points) {
    CHECK_INPUT(sigma);
    CHECK_INPUT(origins);
    CHECK_INPUT(points);
    return argmax_cuda(sigma, origins, points);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("argmax", &argmax, "argmax");
}