add_executable(aip main.cpp)
target_link_libraries(aip PRIVATE aip_core)

# Command-line test tool for TensorRT engines; only built where TensorRT was found.
if(AIP_HAS_TENSORRT)
  add_executable(aip_trt_infer trt_infer.cpp)
  target_link_libraries(aip_trt_infer PRIVATE aip_trt)
endif()
