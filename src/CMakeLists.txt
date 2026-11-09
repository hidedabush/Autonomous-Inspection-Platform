# Core library: pipeline interfaces + CPU baselines (+ CUDA kernels when available)
add_library(aip_core
  core/model_input.cpp
  core/pipeline.cpp
  cpu/grayscale_cpu.cpp
)
target_include_directories(aip_core PUBLIC ${PROJECT_SOURCE_DIR}/include)

if(AIP_HAS_CUDA)
  target_sources(aip_core PRIVATE cuda/grayscale.cu)
  target_compile_definitions(aip_core PUBLIC AIP_HAS_CUDA=1)
  set_target_properties(aip_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# TensorRT classifier: separate library so CPU-only and CI builds don't need TensorRT.
if(AIP_HAS_TENSORRT)
  add_library(aip_trt trt/trt_classifier.cpp)
  # SYSTEM: warnings inside TensorRT's own headers are not ours to fix, so keep them out of /W4.
  # PRIVATE: only trt_classifier.cpp includes them; the public header hides TensorRT (pimpl).
  target_include_directories(aip_trt SYSTEM PRIVATE ${TENSORRT_INCLUDE_DIR})
  # PUBLIC aip_core: users of aip_trt also get aip::Image, Classifier and to_model_input.
  target_link_libraries(aip_trt PUBLIC aip_core PRIVATE ${TENSORRT_LIBRARY} CUDA::cudart)
endif()

add_subdirectory(app)
# add_subdirectory(ui)   # Phase 2: ImGui/OpenGL dashboard
# add_subdirectory(hal)  # Phase 2: camera + serial
