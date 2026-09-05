add_library(parallax_perception
    nanoowl_bridge.cpp
    detection_producer.cpp
    efficientvit_sam.cpp
    segmentation_producer.cpp
    image_space_mapper.cpp
)

find_library(
    TENSORRT_LIBRARY
    NAMES nvinfer
    PATHS /opt/tensorrt-10.4/targets/aarch64-linux-gnu/lib
    NO_DEFAULT_PATH
    REQUIRED
)

target_include_directories(parallax_perception
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include

    PRIVATE
        ${TORCH_INCLUDE_DIRS}
        /opt/tensorrt-10.4/include
)

target_link_libraries(parallax_perception
    PUBLIC
        parallax_core
        parallax_isp
        parallax_cuda
        parallax_stereo
        CUDA::cudart
        ${OpenCV_LIBS}

    PRIVATE
        ${TORCH_LIBRARIES}
        ${TORCH_PYTHON_LIBRARY}
        ${TENSORRT_LIBRARY}
        ${Python3_LIBRARIES}
        Python3::Python
)

target_compile_features(parallax_perception
    PUBLIC
        cxx_std_17
)