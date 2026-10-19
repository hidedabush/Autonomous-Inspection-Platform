# Core library: pipeline interfaces + CPU baselines (+ CUDA kernels when available)
add_library(aip_core
  core/pipeline.cpp
  cpu/grayscale_cpu.cpp
)
target_include_directories(aip_core PUBLIC ${PROJECT_SOURCE_DIR}/include)

if(AIP_HAS_CUDA)
  target_sources(aip_core PRIVATE cuda/grayscale.cu)
  target_compile_definitions(aip_core PUBLIC AIP_HAS_CUDA=1)
  set_target_properties(aip_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

add_subdirectory(app)
# add_subdirectory(ui)   # Phase 2: ImGui/OpenGL dashboard
# add_subdirectory(hal)  # Phase 2: camera + serial
