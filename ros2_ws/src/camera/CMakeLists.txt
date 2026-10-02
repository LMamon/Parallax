cmake_minimum_required(VERSION 3.18)
project(camera LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 87)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)
find_package(
    Torch REQUIRED
    PATHS /usr/local/lib/python3.10/dist-packages/torch/share/cmake
    NO_DEFAULT_PATH
)
find_package(interfaces REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(isaac_ros_managed_nitros REQUIRED)
find_package(isaac_ros_nitros_image_type REQUIRED)

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
  ${ISP_SOURCES}
  ${STEREO_SOURCES}
  ${VPI_SOURCES}
  ${CORE_SOURCES}
  ${CUDA_SOURCES}
    src/perception/nanoowl_bridge.cpp
    src/perception/efficientvit_sam.cpp
    src/perception/perception_runtime.cpp
    src/tracking/dcf_tracker.cpp
    src/perception/efficientvit_sam_cuda.cu
)

target_compile_features(camera PUBLIC cxx_std_17)

set_target_properties(camera PROPERTIES
  CUDA_ARCHITECTURES 87
  POSITION_INDEPENDENT_CODE ON
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
    ${TORCH_LIBRARIES}
    Python3::Python
    pybind11::embed
)

ament_target_dependencies(
  camera
  rclcpp
  sensor_msgs
  interfaces
)

add_library(stereo_component SHARED
  src/node/stereo_node.cpp
  src/node/capture.cpp
  src/node/compute.cpp
  src/node/control.cpp
  src/node/rgb.cpp
  src/node/preview.cpp
)

target_compile_features(stereo_component PRIVATE cxx_std_17)

target_include_directories(stereo_component
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

ament_target_dependencies(
  stereo_component
  rclcpp
  rclcpp_components
  sensor_msgs
  isaac_ros_managed_nitros
  isaac_ros_nitros_image_type
)

target_link_libraries(stereo_component
    camera
)

rclcpp_components_register_nodes(
  stereo_component
  "parallax::ros::StereoNode"
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS
    camera
    stereo_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_package()