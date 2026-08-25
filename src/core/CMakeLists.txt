add_library(parallax_core STATIC
    dependency_resolver.cpp
    graph.cpp
    pipeline.cpp
    product_store.cpp
    runtime.cpp
)

target_include_directories(parallax_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(parallax_core
    PUBLIC
        parallax_camera
        parallax_isp
        parallax_lidar
        parallax_pose
        parallax_stereo
        parallax_vpi
        parallax_visualization
)

target_compile_features(parallax_core 
    PUBLIC 
        cxx_std_17
)