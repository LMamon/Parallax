add_library(parallax_stereo
    calibration.cpp
    rectification.cpp
    rectification_producer.cpp
    stereo_producer.cpp
    matcher.cpp
)

target_include_directories(parallax_stereo
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(parallax_stereo
    PUBLIC
        parallax_cuda
        parallax_vpi
)

target_compile_features(parallax_stereo
    PUBLIC
        cxx_std_17
)