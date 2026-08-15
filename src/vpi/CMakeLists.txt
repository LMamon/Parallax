add_library(parallax_vpi
    image_wrapper.cpp
    stream.cpp
)

target_include_directories(parallax_vpi
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(parallax_vpi
    PUBLIC
        parallax_cuda
        VPI::VPI
)
