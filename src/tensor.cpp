#include "tensor.h"

#include <stdexcept>
#include <sstream>

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xpad.hpp>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

namespace library {
    void cudnn_check(int line, cudnnStatus_t status) {
        if (status != CUDNN_STATUS_SUCCESS) {
            std::stringstream message;
            message << "cuDNN at line: " << line << ", status = " << status;
            throw std::runtime_error(message.str());
        }
    }

    void cuda_check(int line, cudaError_t error) {
        if (error != cudaSuccess) {
            std::stringstream message;
            message << "CUDA at line: " << line << ", error = " << error;
            throw std::runtime_error(message.str());
        }
    }
}

using namespace library;

Image::Image(const xt::pyarray<float>& image) {
    cudnn_check(__LINE__, cudnnCreateTensorDescriptor(&m_descriptor));

    int height = static_cast<int>(image.shape(0));
    int width = static_cast<int>(image.shape(1));
    int channels = static_cast<int>(image.shape(2));

    cudnn_check(__LINE__, cudnnSetTensor4dDescriptor(
        m_descriptor,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        1,
        channels,
        height,
        width
    ));

    m_data_size = channels * height * width * sizeof(float);
    cuda_check(__LINE__, cudaMalloc(&m_data, m_data_size));
    cudaMemcpy(m_data, image.data(), m_data_size, cudaMemcpyHostToDevice);
}

Image::Image(int height, int width, int channels) {
    cudnn_check(__LINE__, cudnnCreateTensorDescriptor(&m_descriptor));

    cudnn_check(__LINE__, cudnnSetTensor4dDescriptor(
        m_descriptor,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        1,
        channels,
        height,
        width
    ));

    m_data_size = channels * height * width * sizeof(float);
    cuda_check(__LINE__, cudaMalloc(&m_data, m_data_size));
    cudaMemset(m_data, 0, m_data_size);
}

Image::~Image() {
    if (m_data != nullptr) {
        cudaFree(m_data);
    }

    if (m_descriptor != nullptr) {
        cudnnDestroyTensorDescriptor(m_descriptor);
    }
}

cudnnTensorDescriptor_t Image::descriptor() const {
    return m_descriptor;
}

float* Image::data() const {
    return m_data;
}

void Image::copy(xt::xarray<float>& target) const {
    cudaMemcpy(target.data(), m_data, m_data_size, cudaMemcpyDeviceToHost);
}

Kernel::Kernel(int channels, int kernel_size, float sigma) : m_group_count(channels) {
    cudnn_check(__LINE__, cudnnCreateFilterDescriptor(&m_descriptor));

    cudnn_check(__LINE__, cudnnSetFilter4dDescriptor(
        m_descriptor,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        channels,
        channels / m_group_count,
        kernel_size,
        kernel_size
    ));

    std::array<int, 2> shape{kernel_size, kernel_size};
    xt::xarray<float> x = xt::empty<float>(shape);
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            x(i, j) = static_cast<float>(j);
        }
    }

    xt::xarray<float> y = xt::transpose(x);
    auto xy = xt::stack(xt::xtuple(x, y), 1);

    float mean = (kernel_size - 1.0f) / 2.0f;
    float variance = sigma * sigma;

    auto t1 = xt::pow((xy - mean), 2.0);
    xt::xarray<float> k1 = (1.0 / (2.0 * M_PI * variance)) * xt::exp(-xt::sum(t1, 1) / (2.0 * variance));
    xt::xarray<float> k2 = k1 / xt::sum(k1);

    auto r1 = xt::view(k2, xt::newaxis(), xt::newaxis(), xt::all(), xt::all());
    auto r2 = xt::tile(r1, {channels, 1, 1, 1});

    size_t size = r2.size() * sizeof(float);
    cuda_check(__LINE__, cudaMalloc(&m_data, size));
    cudaMemcpy(m_data, r2.data(), size, cudaMemcpyHostToDevice);
}

Kernel::~Kernel() {
    if (m_data != nullptr) {
        cudaFree(m_data);
    }

    if (m_descriptor != nullptr) {
        cudnnDestroyFilterDescriptor(m_descriptor);
    }
}

int Kernel::group_count() const {
    return m_group_count;
}

cudnnFilterDescriptor_t Kernel::descriptor() const {
    return m_descriptor;
}

float* Kernel::data() const {
    return m_data;
}

Operation::Operation() {
    cudnn_check(__LINE__, cudnnCreateConvolutionDescriptor(&m_descriptor));

    cudnn_check(__LINE__, cudnnSetConvolution2dDescriptor(
        m_descriptor,
        0,
        0,
        1,
        1,
        1,
        1,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    ));
}

Operation::~Operation() {
    if (m_workspace != nullptr) {
        cudaFree(m_workspace);
    }

    if (m_descriptor != nullptr) {
        cudnnDestroyConvolutionDescriptor(m_descriptor);
    }
}


xt::pyarray<float> Operation::operator()(cudnnHandle_t handle, const Image& input, const Kernel& kernel) {
    cudnn_check(__LINE__, cudnnSetConvolutionGroupCount(m_descriptor, kernel.group_count()));

    int temp = 0;
    int out_height = 0;
    int out_width = 0;
    int out_channels = 0;
    cudnn_check(__LINE__, cudnnGetConvolution2dForwardOutputDim(
        m_descriptor,
        input.descriptor(),
        kernel.descriptor(),
        &temp,
        &out_channels,
        &out_height,
        &out_width
    ));

    Image output{out_height, out_width, out_channels};

    cudnnConvolutionFwdAlgo_t algorithm;
    cudnn_check(__LINE__, cudnnGetConvolutionForwardAlgorithm(
        handle,
        input.descriptor(),
        kernel.descriptor(),
        m_descriptor,
        output.descriptor(),
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &algorithm
    ));

    cudnn_check(__LINE__, cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        input.descriptor(),
        kernel.descriptor(),
        m_descriptor,
        output.descriptor(),
        algorithm,
        &m_workspace_size
    ));

    cuda_check(__LINE__, cudaMalloc(&m_workspace, m_workspace_size));

    const float alpha = 1;
    const float beta = 0;
    cudnn_check(__LINE__, cudnnConvolutionForward(
        handle,
        &alpha,
        input.descriptor(),
        input.data(),
        kernel.descriptor(),
        kernel.data(),
        m_descriptor,
        algorithm,
        m_workspace,
        m_workspace_size,
        &beta,
        output.descriptor(),
        output.data()
    ));

    auto result = xt::empty<float>(std::vector<int>{out_height, out_width, out_channels});
    output.copy(result);

    return result;
}
