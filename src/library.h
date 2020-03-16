#pragma once

#include <exception>

#include <xtensor-python/pyarray.hpp>

class ImageProcessingLibrary final {
public:
    ImageProcessingLibrary();
    ~ImageProcessingLibrary();

    xt::pyarray<float> run(const xt::pyarray<float>& image, int kernel_size, float sigma);

private:
    struct Context;
    Context *m_context = nullptr;
};
