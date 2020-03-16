#include "library.h"

#include <stdexcept>

#include <cudnn.h>

#include "tensor.h"

struct ImageProcessingLibrary::Context {
    cudnnHandle_t handle = nullptr;
};

ImageProcessingLibrary::ImageProcessingLibrary() : m_context(new Context()) {
    cudnnStatus_t status = cudnnCreate(&(m_context->handle));
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("cudnnCreate");
    }
}

ImageProcessingLibrary::~ImageProcessingLibrary() {
    if (m_context != nullptr) {
        if (m_context->handle != nullptr) {
            cudnnDestroy(m_context->handle);
        }
        delete m_context;
    }
}

xt::pyarray<float> ImageProcessingLibrary::run(const xt::pyarray<float>& image, int kernel_size, float sigma) {
    if (image.dimension() != 3) {
        throw std::runtime_error("wrong image dimension");
    }

    library::Image input{image};

    int channels = static_cast<int>(image.shape(2));
    library::Kernel kernel{channels, kernel_size, sigma};

    library::Operation operation;
    return operation(m_context->handle, input, kernel);
}
