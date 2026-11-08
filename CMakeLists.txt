cmake_minimum_required(VERSION 3.24)
project(aip VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(AIP_ENABLE_CUDA "Build CUDA kernels if a CUDA compiler is found" ON)
option(AIP_BUILD_TESTS "Build unit tests" ON)

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
