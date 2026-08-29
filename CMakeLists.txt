cmake_minimum_required(VERSION 3.18)

project(
    Parallax
    VERSION 2.0
    LANGUAGES CXX CUDA
)
set(CMAKE_CUDA_ARCHITECTURES 87)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)
include(FetchContent)

FetchContent_Declare(
    foxglove_sdk
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    URL "https://github.com/foxglove/foxglove-sdk/releases/download/sdk%2Fv0.26.0/foxglove-v0.26.0-cpp-aarch64-unknown-linux-gnu.zip"
    URL_HASH SHA256=0710f2ec5abc3954acf6203b93a96768dccd19d742af9b78fcd8fbbba5f6225a
)

FetchContent_MakeAvailable(foxglove_sdk)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
    )
endif()

find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(
    foxglove-sdk
    CONFIG REQUIRED
    HINTS "${foxglove_sdk_SOURCE_DIR}"
)

foxglove_sdk_add_cpp_library(foxglove_cpp TYPE STATIC)

find_path(VPI_INCLUDE_DIR
    NAMES vpi/Image.h
    PATHS /opt/nvidia/vpi3/include
    REQUIRED
)

find_library(VPI_LIBRARY
    NAMES nvvpi
    PATHS /opt/nvidia/vpi3/lib/aarch64-linux-gnu
    REQUIRED
)

add_library(VPI::VPI UNKNOWN IMPORTED)

set_target_properties(VPI::VPI PROPERTIES
    IMPORTED_LOCATION "${VPI_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${VPI_INCLUDE_DIR}"
)

add_subdirectory(src/application)
add_subdirectory(src/camera)
add_subdirectory(src/core)
add_subdirectory(src/cuda)
add_subdirectory(src/isp)
add_subdirectory(src/lidar)
add_subdirectory(src/stereo)
add_subdirectory(src/pose)
add_subdirectory(src/visualization)
add_subdirectory(src/vpi)

# Add these when their CMakeLists.txt files exist.
# add_subdirectory(src/bindings)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

add_executable(parallax
    src/main.cpp
)

target_compile_definitions(parallax_visualization
    PRIVATE
        PARALLAX_SCHEMA_DIR="${CMAKE_SOURCE_DIR}/schemas"
)

target_include_directories(parallax
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(parallax
    PRIVATE
        parallax_camera
        parallax_core
        parallax_application
        parallax_cuda
        parallax_pose
        parallax_isp
        parallax_stereo
        parallax_visualization
        parallax_vpi
        ${OpenCV_LIBS}
)