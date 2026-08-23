add_library(parallax_cuda
    cuda_buffer.cu
    depth.cu
)

target_include_directories(parallax_cuda
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(parallax_cuda
    PUBLIC
        CUDA::cudart
)

target_compile_features(parallax_cuda
    PUBLIC
        cxx_std_17
        cuda_std_17
)

set_target_properties(parallax_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)