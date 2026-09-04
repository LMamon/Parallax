add_library(parallax_tracking
    dcf_tracker.cpp
)

target_include_directories(parallax_tracking
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(parallax_tracking
    PUBLIC
        parallax_cuda
        parallax_vpi
        VPI::VPI
        ${OpenCV_LIBS}
)