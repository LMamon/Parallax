add_library(parallax_application
    command_parser.cpp
    foxglove_command.cpp
    request_controller.cpp
)

target_include_directories(parallax_application
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(parallax_application
    PUBLIC
        cxx_std_17
)

target_link_libraries(parallax_application
    PUBLIC
        parallax_core
)