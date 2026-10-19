# Plain assert-based test to start; switch to GoogleTest (FetchContent) in Phase 2.
add_executable(test_grayscale test_grayscale.cpp)
target_link_libraries(test_grayscale PRIVATE aip_core)
add_test(NAME grayscale COMMAND test_grayscale)
