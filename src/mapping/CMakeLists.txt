add_library(parallax_mapping
    local_occupancy_producer.cpp
    occupancy_snapshot.cpp
)

target_include_directories(parallax_mapping
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(parallax_mapping
    PUBLIC
        parallax_core
        parallax_stereo
        parallax_localization
        nvblox::nvblox_lib
        CUDA::cudart
)

target_compile_features(parallax_mapping PUBLIC cxx_std_17)
