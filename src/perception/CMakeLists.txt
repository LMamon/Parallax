add_library(parallax_perception INTERFACE)

target_include_directories(parallax_perception
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(parallax_perception
    INTERFACE
        ${OpenCV_LIBS}
)

target_compile_features(parallax_perception
    INTERFACE
        cxx_std_17
)