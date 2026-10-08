cmake_minimum_required(VERSION 3.18)
project(perception LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 87)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(interfaces REQUIRED)
find_package(camera REQUIRED)
find_package(isaac_ros_managed_nitros REQUIRED)
find_package(isaac_ros_nitros_image_type REQUIRED)

find_package(pybind11 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)

find_package(
  Torch REQUIRED
  PATHS /usr/local/lib/python3.10/dist-packages/torch/share/cmake
  NO_DEFAULT_PATH
)

find_library(TORCH_PYTHON_LIBRARY
  NAMES torch_python
  PATHS /usr/local/lib/python3.10/dist-packages/torch/lib
  REQUIRED
)

find_package(OpenCV REQUIRED COMPONENTS core)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(vpi 3.2 REQUIRED)

add_library(parallax_perception STATIC
  src/perception/nanoowl_bridge.cpp
  src/perception/efficientvit_sam.cpp
  src/perception/efficientvit_sam_cuda.cu
  src/perception/perception_runtime.cpp
  src/tracking/dcf_tracker.cpp
)

target_compile_features(parallax_perception PUBLIC cxx_std_17)

set_target_properties(parallax_perception PROPERTIES
  CUDA_ARCHITECTURES 87
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(parallax_perception
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Keep target_link_libraries in plain-signature form because
# ament_target_dependencies() on ROS 2 Humble also uses the plain signature.
target_link_libraries(parallax_perception
  camera::camera
  CUDA::cudart
  Threads::Threads
  vpi
  dl
  ${OpenCV_LIBS}
  ${TORCH_LIBRARIES}
  ${TORCH_PYTHON_LIBRARY}
  nvinfer
  Python3::Python
  pybind11::embed
)

ament_target_dependencies(parallax_perception
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

ament_target_dependencies(stereo_component
  rclcpp
  rclcpp_components
  sensor_msgs
  isaac_ros_managed_nitros
  isaac_ros_nitros_image_type
)

target_link_libraries(stereo_component
  parallax_perception
  camera::camera
)

rclcpp_components_register_nodes(
  stereo_component
  "parallax::ros::StereoNode"
)

install(
  DIRECTORY include/
  DESTINATION include
)

ament_python_install_package(parallax_perception
  PACKAGE_DIR python/parallax_perception
)

install(
  TARGETS parallax_perception
  EXPORT export_parallax_perception
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(
  TARGETS stereo_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_export_targets(
  export_parallax_perception
  HAS_LIBRARY_TARGET
)

ament_export_dependencies(
  rclcpp
  sensor_msgs
  interfaces
  camera
)

ament_package()