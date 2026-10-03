cmake_minimum_required(VERSION 3.18)
project(camera LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 87)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(vpi 3.2 REQUIRED)

file(GLOB CAMERA_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/camera/*.cpp"
)

file(GLOB CORE_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp"
)

file(GLOB ISP_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/isp/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/isp/*.cu"
)

file(GLOB STEREO_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/stereo/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/stereo/*.cu"
)

file(GLOB VPI_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/vpi/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/vpi/*.cu"
)

set(CUDA_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/cuda_buffer.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/depth.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/gray_resize.cu"
)

add_library(camera STATIC
  ${CAMERA_SOURCES}
  ${CORE_SOURCES}
  ${ISP_SOURCES}
  ${STEREO_SOURCES}
  ${VPI_SOURCES}
  ${CUDA_SOURCES}
)

add_library(camera::camera ALIAS camera)

target_compile_features(camera PUBLIC cxx_std_17)

set_target_properties(camera PROPERTIES
  CUDA_ARCHITECTURES 87
  POSITION_INDEPENDENT_CODE ON
  EXPORT_NAME camera
)

target_include_directories(camera
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(camera
  CUDA::cudart
  CUDA::nppicc
  yaml-cpp
  Threads::Threads
  vpi
  dl
  ${OpenCV_LIBS}
)

ament_target_dependencies(
  camera
  rclcpp
  sensor_msgs
)

# StereoNode is built by the perception package because it owns the
# in-process PerceptionRuntime. Camera exports only the sensor/core
# implementation library.

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS camera
  EXPORT export_camera
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

ament_export_targets(
  export_camera
  HAS_LIBRARY_TARGET
  NAMESPACE camera::
)

ament_export_dependencies(
  rclcpp
  sensor_msgs
)

ament_package()