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

add_subdirectory(src/camera)
add_subdirectory(src/cuda)
add_subdirectory(src/vpi)
add_subdirectory(src/isp)
add_subdirectory(src/stereo)

# Add these when their CMakeLists.txt files exist.
# add_subdirectory(src/bindings)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

add_executable(parallax
    src/main.cpp
)

target_include_directories(parallax
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(parallax
    PRIVATE
        parallax_camera
        parallax_cuda
        parallax_isp
        parallax_stereo
        parallax_vpi
        ${OpenCV_LIBS}
)