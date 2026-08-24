add_library(parallax_pose
    charuco_pose.cpp
    charuco_pose_producer.cpp
    marker_depth_producer.cpp
)

find_package(OpenCV REQUIRED)

target_include_directories(parallax_pose
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(parallax_pose
    PUBLIC
        ${OpenCV_LIBS}
        CUDA::cudart
)

target_compile_features(parallax_pose
    PUBLIC
        cxx_std_17
)