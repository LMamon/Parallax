find_package(GTest REQUIRED)
include(GoogleTest)

function(parallax_add_gtest target source)
    add_executable(${target} ${source})

    target_link_libraries(${target}
        PRIVATE
            GTest::gtest_main
            ${ARGN}
    )

    gtest_discover_tests(${target})
endfunction()

parallax_add_gtest(
    parallax_tests
    smoke_test.cpp
    parallax_core
)

parallax_add_gtest(
    parallax_product_store_test
    product_store_test.cpp
    parallax_core
)

parallax_add_gtest(
    parallax_graph_test
    graph_test.cpp
    parallax_core
)

parallax_add_gtest(
    parallax_runtime_metrics_test
    runtime_metrics_test.cpp
    parallax_core
    parallax_cuda
    CUDA::cudart
)

parallax_add_gtest(
    parallax_request_controller_test
    request_controller_test.cpp
    parallax_application
    parallax_core
)

parallax_add_gtest(
    parallax_detection_test
    detection_test.cpp
    parallax_core
    parallax_perception
)

parallax_add_gtest(
    parallax_segmentation_test
    segmentation_test.cpp
    parallax_core
    parallax_perception
)

parallax_add_gtest(
    parallax_object3d_test
    object3d_test.cpp
    parallax_core
    parallax_perception
)

parallax_add_gtest(
    parallax_object3d_association_test
    object3d_association_test.cpp
    parallax_core
    parallax_perception
)

parallax_add_gtest(
    parallax_tracking_test
    tracking_test.cpp
    parallax_tracking
)

parallax_add_gtest(
    parallax_dcf_tracker_test
    dcf_tracker_test.cpp
    parallax_tracking
)

add_executable(
    parallax_rplidar_smoke
    rplidar_smoke.cpp
)

target_link_libraries(parallax_rplidar_smoke
    PRIVATE
        parallax_lidar
)

add_executable(
    parallax_efficientvit_sam_smoke
    efficientvit_sam_smoke.cpp
)

target_link_libraries(parallax_efficientvit_sam_smoke
    PRIVATE
        parallax_perception
)