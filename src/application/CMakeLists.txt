add_library(parallax_application
    command_parser.cpp
)

target_include_directories(parallax_application
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(parallax_application
    PUBLIC
        cxx_std_17
)