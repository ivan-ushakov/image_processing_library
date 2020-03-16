#pragma once

#include <cudnn.h>

#include <xtensor-python/pyarray.hpp>

namespace library {
    class Image final {
    public:
        explicit Image(const xt::pyarray<float>& image);
        Image(int height, int width, int channels);
        ~Image();

        Image(const Image&) = delete;
        Image& operator=(const Image&) = delete;

        cudnnTensorDescriptor_t descriptor() const;
        float* data() const;

        void copy(xt::xarray<float>& target) const;

    private:
        cudnnTensorDescriptor_t m_descriptor = nullptr;
        size_t m_data_size = 0;
        float* m_data = nullptr;
    };

    class Kernel final {
    public:
        Kernel(int channels, int kernel_size, float sigma);
        ~Kernel();

        Kernel(const Kernel&) = delete;
        Kernel& operator=(const Kernel&) = delete;

        int group_count() const;

        cudnnFilterDescriptor_t descriptor() const;
        float* data() const;

    private:
        int m_group_count = 0;
        cudnnFilterDescriptor_t m_descriptor = nullptr;
        float* m_data = nullptr;
    };

    class Operation final {
    public:
        explicit Operation();
        ~Operation();

        Operation(const Operation&) = delete;
        Operation& operator=(const Operation&) = delete;

        xt::pyarray<float> operator()(cudnnHandle_t handle, const Image& input, const Kernel& kernel);

    private:
        cudnnConvolutionDescriptor_t m_descriptor = nullptr;
        size_t m_workspace_size = 0;
        void* m_workspace = nullptr;
    };
}
