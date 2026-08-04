cmake_minimum_required(VERSION 3.18)

project(
    Parallax
    VERSION 2.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_language(CUDA)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_subdirectory(src/camera)

add_subdirectory(src/isp)
add_subdirectory(src/stereo)
add_subdirectory(src/bindings)

add_library(parallax_camera
    src/camera/arducam_device.cpp
    src/camera/camera_config.cpp
    src/camera/logger.cpp
    src/camera/stereo_camera.cpp
    src/camera/v4l2_device.cpp
)

add_executable(parallax
    src/main.cpp
)

target_link_libraries(parallax
PRIVATE
    parallax_camera
    parallax_isp
    parallax_stereo
)