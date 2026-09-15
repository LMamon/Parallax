add_library(parallax_isp
    isp.cpp
    isp_config.cpp
    isp_producer.cpp
    auto_control.cpp
    isp_kernels.cu
)

target_include_directories(parallax_isp
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(parallax_isp
    PUBLIC
        parallax_camera
        parallax_cuda
        CUDA::cudart
        CUDA::nppicc
        yaml-cpp
)

target_compile_features(parallax_isp
    PUBLIC
        cxx_std_17
        cuda_std_17
)

set_target_properties(parallax_isp PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)