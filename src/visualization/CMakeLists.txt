add_library(parallax_visualization
    foxglove_server.cpp
    publisher.cpp
)

target_include_directories(parallax_visualization
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(parallax_visualization
    PUBLIC foxglove_cpp parallax_pose CUDA::cudart
    PRIVATE nlohmann_json::nlohmann_json ${OpenCV_LIBS}
)

target_compile_features(parallax_visualization PUBLIC cxx_std_17)
