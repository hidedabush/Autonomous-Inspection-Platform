cmake_minimum_required(VERSION 3.24)
project(aip VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(AIP_ENABLE_CUDA "Build CUDA kernels if a CUDA compiler is found" ON)
option(AIP_BUILD_TESTS "Build unit tests" ON)
option(AIP_ENABLE_TENSORRT "Build the TensorRT classifier if TensorRT is found" ON)

set(AIP_HAS_CUDA OFF)
if(AIP_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # Set to your GPU (e.g. 86 for RTX 30xx, 89 for RTX 40xx). Jetson Nano = 53, Orin = 87.
    # Must come before enable_language(CUDA), which otherwise picks the compiler's default.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(AIP_HAS_CUDA ON)
    message(STATUS "AIP: CUDA enabled (${CMAKE_CUDA_COMPILER_VERSION})")
  else()
    message(STATUS "AIP: no CUDA compiler found, building CPU-only")
  endif()
endif()

# TensorRT only needs the CUDA runtime, not nvcc. Point TENSORRT_ROOT at the install if it is not
# in a system path (e.g. the extracted TensorRT zip on Windows); JetPack installs it system-wide.
# Optional by design: when anything is missing, CMake prints why and builds everything else, so
# CI and CPU-only laptops are unaffected. AIP_HAS_TENSORRT gates the targets in src/.
set(AIP_HAS_TENSORRT OFF)
if(AIP_ENABLE_TENSORRT)
  # Provides the CUDA::cudart target (the CUDA runtime library TensorRT code calls).
  find_package(CUDAToolkit QUIET)
  # TENSORRT_ROOT may be given as -DTENSORRT_ROOT=... or as an environment variable.
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_ROOT} ENV TENSORRT_ROOT PATH_SUFFIXES include)
  # The library is libnvinfer.so on Linux/Jetson and nvinfer_10.lib in the TensorRT 10 Windows zip.
  find_library(TENSORRT_LIBRARY NAMES nvinfer nvinfer_10 HINTS ${TENSORRT_ROOT} ENV TENSORRT_ROOT
               PATH_SUFFIXES lib lib64)
  if(CUDAToolkit_FOUND AND TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
    set(AIP_HAS_TENSORRT ON)
    message(STATUS "AIP: TensorRT enabled (${TENSORRT_LIBRARY})")
  else()
    message(STATUS "AIP: TensorRT not found, skipping the TensorRT classifier (set TENSORRT_ROOT)")
  endif()
endif()

# Warnings for C++ only; nvcc doesn't accept these flags directly.
if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall$<SEMICOLON>-Wextra>)
endif()

add_subdirectory(src)

if(AIP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
