set(SLAMTEC_RPLIDAR_SDK_ROOT
    "$ENV{HOME}/rplidar_sdk"
    CACHE PATH
    "Path to the SLAMTEC RPLIDAR SDK")

set(SLAMTEC_RPLIDAR_INCLUDE_DIR
    "${SLAMTEC_RPLIDAR_SDK_ROOT}/sdk/include")

set(SLAMTEC_RPLIDAR_LIBRARY
    "${SLAMTEC_RPLIDAR_SDK_ROOT}/output/Linux/Release/libsl_lidar_sdk.a")

if(NOT EXISTS "${SLAMTEC_RPLIDAR_INCLUDE_DIR}/sl_lidar.h")
    message(FATAL_ERROR
        "SLAMTEC RPLIDAR SDK headers not found at "
        "${SLAMTEC_RPLIDAR_INCLUDE_DIR}")
endif()

if(NOT EXISTS "${SLAMTEC_RPLIDAR_LIBRARY}")
    message(FATAL_ERROR
        "SLAMTEC RPLIDAR SDK library not found at "
        "${SLAMTEC_RPLIDAR_LIBRARY}. Build the SDK before configuring Parallax.")
endif()

add_library(parallax_lidar
    rplidar.cpp
    rplidar_producer.cpp
)

target_include_directories(parallax_lidar
    PUBLIC
        parallax_core
        ${PROJECT_SOURCE_DIR}/include

    PRIVATE
        ${SLAMTEC_RPLIDAR_INCLUDE_DIR}
)

target_link_libraries(parallax_lidar
    PRIVATE
        ${SLAMTEC_RPLIDAR_LIBRARY}
        pthread
)

target_compile_options(parallax_lidar
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)