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
)

target_compile_features(parallax_perception
    PUBLIC
        cxx_std_17
)