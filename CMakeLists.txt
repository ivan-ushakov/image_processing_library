cmake_minimum_required(VERSION 3.10)
project(image_processing_library)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT DEFINED CUDA_PATH)
  message("CUDA_PATH is not defined")
  return()
endif()

# set(CUDA_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2")
include_directories(${CUDA_PATH}/include)
link_directories(${CUDA_PATH}/lib/x64)

if(NOT DEFINED CDNN_PATH)
  message("CDNN_PATH is not defined")
  return()
endif()

# set(CDNN_PATH "C:/Tools/cudnn-10.2-windows10-x64-v7.6.5.32")
include_directories(${CDNN_PATH}/include)
link_directories(${CDNN_PATH}/lib/x64)

if(NOT DEFINED XTL_PATH)
  message("XTL_PATH is not defined")
  return()
endif()

# set(XTL_PATH "C:/Tools/xtl-0.6.13")
include_directories(${XTL_PATH}/include)

if(NOT DEFINED XTENSOR_PATH)
  message("XTENSOR_PATH is not defined")
  return()
endif()

# set(XTENSOR_PATH "C:/Tools/xtensor-0.21.4")
include_directories(${XTENSOR_PATH}/include)

if(NOT DEFINED XTENSOR_PYTHON_PATH)
  message("XTENSOR_PYTHON_PATH is not defined")
  return()
endif()

# set(XTENSOR_PYTHON_PATH "C:/Tools/xtensor-python-0.24.1")
include_directories(${XTENSOR_PYTHON_PATH}/include)

if(NOT DEFINED NUMPY_CORE_PATH)
  message("NUMPY_CORE_PATH is not defined")
  return()
endif()

# set(NUMPY_CORE_PATH "C:/Users/Ivan/Anaconda3/lib/site-packages/numpy/core")
include_directories(${NUMPY_CORE_PATH}/include)

add_subdirectory(pybind11)

set(SOURCE_LIB src/tensor.h src/tensor.cpp src/library.h src/library.cpp src/main.cpp)

add_library(image_processing_library MODULE ${SOURCE_LIB})
target_link_libraries(image_processing_library PRIVATE pybind11::module cudart cudnn)

set_target_properties(image_processing_library PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")
