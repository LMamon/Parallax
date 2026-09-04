add_library(parallax_runtime STATIC
    ../runtime.cpp
)

target_include_directories(parallax_runtime
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(parallax_runtime
    PUBLIC
        parallax_core
        parallax_application
        parallax_camera
        parallax_isp
        parallax_lidar
        parallax_perception
        parallax_pose
        parallax_stereo
        parallax_tracking
        parallax_visualization
        parallax_vpi
)

target_compile_features(parallax_runtime
    PUBLIC
        cxx_std_17
)