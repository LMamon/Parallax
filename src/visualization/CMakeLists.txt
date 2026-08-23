add_library(parallax_visualization
    foxglove_server.cpp
    publisher.cpp
    video_encoder.cpp
)

find_package(PkgConfig REQUIRED)

pkg_check_modules(
    GSTREAMER
    REQUIRED
    IMPORTED_TARGET
    gstreamer-1.0
    gstreamer-app-1.0
)

target_include_directories(parallax_visualization
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(parallax_visualization
    PUBLIC
        foxglove_cpp
        parallax_pose
        CUDA::cudart

    PRIVATE
        PkgConfig::GSTREAMER
)

target_compile_features(parallax_visualization
    PUBLIC
        cxx_std_17
)