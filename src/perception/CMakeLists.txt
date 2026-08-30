find_package(
    Python3
    REQUIRED
    COMPONENTS
        Interpreter
        Development
)

find_package(
    Torch
    REQUIRED
    PATHS
        /usr/local/lib/python3.10/dist-packages/torch/share/cmake
    NO_DEFAULT_PATH
)

find_library(
    TORCH_PYTHON_LIBRARY
    NAMES torch_python
    PATHS
        /usr/local/lib/python3.10/dist-packages/torch/lib
    REQUIRED
    NO_DEFAULT_PATH
)

add_library(parallax_perception
    nanoowl_bridge.cpp
    detection_producer.cpp
)

target_include_directories(parallax_perception
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include

    PRIVATE
        ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(parallax_perception
    PUBLIC
        parallax_isp
        CUDA::cudart
        ${OpenCV_LIBS}

    PRIVATE
        ${TORCH_LIBRARIES}
        ${TORCH_PYTHON_LIBRARY}
        Python3::Python
        pybind11::embed
        Torch::Torch
)

target_compile_features(parallax_perception
    PUBLIC
        cxx_std_17
)

set_property(
    TARGET parallax_perception
    PROPERTY
        CXX_STANDARD 17
)