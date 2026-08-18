add_library(parallax_visualization
    foxglove_server.cpp
)

target_include_directories(parallax_visualization
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(parallax_visualization
    PUBLIC
        foxglove_cpp
)

target_compile_features(parallax_visualization
    PUBLIC
        cxx_std_17
)