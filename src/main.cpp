#include <pybind11/pybind11.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>

#include "library.h"

namespace py = pybind11;

PYBIND11_MODULE(image_processing_library, m) {
    xt::import_numpy();

    py::class_<ImageProcessingLibrary>(m, "ImageProcessingLibrary")
        .def(py::init<>())
        .def("run", &ImageProcessingLibrary::run);
}
