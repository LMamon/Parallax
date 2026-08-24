add_library(parallax_isp
    isp.cpp
    isp_producer.cpp
    demosaic.cu
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
)

target_compile_features(parallax_isp
    PUBLIC
        cxx_std_17
        cuda_std_17
)

set_target_properties(parallax_isp PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)