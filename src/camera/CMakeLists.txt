add_library(parallax_camera
    arducam_device.cpp
    camera_config.cpp
    camera_producer.cpp
    logger.cpp
    stereo_camera.cpp
    v4l2_device.cpp
)

target_include_directories(parallax_camera
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(parallax_camera
    PRIVATE
        yaml-cpp
        Threads::Threads
)

target_compile_features(parallax_camera
    PUBLIC
        cxx_std_17
)