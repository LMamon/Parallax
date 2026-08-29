find_package(GTest REQUIRED)

add_executable(parallax_tests
    smoke_test.cpp)

add_executable(parallax_product_store_test
    product_store_test.cpp
)

add_executable(parallax_graph_test
    graph_test.cpp
)

add_executable(parallax_rplidar_smoke
    rplidar_smoke.cpp
)

add_executable(parallax_runtime_metrics_test
    runtime_metrics_test.cpp
)

add_executable(parallax_request_controller_test
    request_controller_test.cpp
)

target_link_libraries(parallax_request_controller_test
    PRIVATE
        parallax_application
        parallax_core
        GTest::gtest_main
)

target_link_libraries(parallax_rplidar_smoke
    PRIVATE
        parallax_lidar
)

target_link_libraries(parallax_runtime_metrics_test
    PRIVATE
        parallax_core
        parallax_cuda
        CUDA::cudart
        GTest::gtest_main
)

target_link_libraries(parallax_tests
    PRIVATE
        parallax_core
        GTest::gtest_main
)

target_link_libraries(parallax_product_store_test
    PRIVATE
        parallax_core
        GTest::gtest_main
)

target_link_libraries(parallax_graph_test
    PRIVATE
        parallax_core
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(parallax_tests)
gtest_discover_tests(parallax_product_store_test)
gtest_discover_tests(parallax_graph_test)
gtest_discover_tests(parallax_runtime_metrics_test)
gtest_discover_tests(parallax_request_controller_test)