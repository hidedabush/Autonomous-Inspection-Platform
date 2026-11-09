# Plain assert-based test to start; switch to GoogleTest (FetchContent) in Phase 2.
add_executable(test_grayscale test_grayscale.cpp)
target_link_libraries(test_grayscale PRIVATE aip_core)
add_test(NAME grayscale COMMAND test_grayscale)

# Preprocessing for the model (resize + layout); needs no GPU, so it runs in CI.
add_executable(test_model_input test_model_input.cpp)
target_link_libraries(test_model_input PRIVATE aip_core)
add_test(NAME model_input COMMAND test_model_input)
