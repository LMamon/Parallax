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

# Add OpenCV when the first component genuinely needs it.
# find_package(OpenCV REQUIRED)

add_subdirectory(src/camera)
add_subdirectory(src/cuda)

# Add these when their CMakeLists.txt files exist.
# add_subdirectory(src/isp)
# add_subdirectory(src/stereo)
# add_subdirectory(src/bindings)

add_executable(parallax
    src/main.cpp
)

target_link_libraries(parallax
    PRIVATE
        parallax_camera
        parallax_cuda
)