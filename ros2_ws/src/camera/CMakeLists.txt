cmake_minimum_required(VERSION 3.18)
project(camera LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(isaac_ros_managed_nitros REQUIRED)
find_package(isaac_ros_nitros_image_type REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(vpi 3.2 REQUIRED)

set(PARALLAX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(PARALLAX_BUILD "${PARALLAX_ROOT}/build")

set(INHERITED_LIBS
  "${PARALLAX_BUILD}/src/stereo/libparallax_stereo.a"
  "${PARALLAX_BUILD}/src/isp/libparallax_isp.a"
  "${PARALLAX_BUILD}/src/camera/libparallax_camera.a"
  "${PARALLAX_BUILD}/src/vpi/libparallax_vpi.a"
  "${PARALLAX_BUILD}/src/cuda/libparallax_cuda.a"
  "${PARALLAX_BUILD}/src/core/libparallax_core.a"
)

foreach(lib IN LISTS INHERITED_LIBS)
  if(NOT EXISTS "${lib}")
    message(FATAL_ERROR
      "Missing inherited library: ${lib}\n"
      "Build the compatibility targets before the ROS workspace.")
  endif()
endforeach()

add_library(stereo_component SHARED src/stereo_node.cpp)
target_compile_features(stereo_component PRIVATE cxx_std_17)
target_include_directories(stereo_component PRIVATE "${PARALLAX_ROOT}/include")
ament_target_dependencies(
  stereo_component
  rclcpp
  rclcpp_components
  sensor_msgs
  isaac_ros_managed_nitros
  isaac_ros_nitros_image_type
)

target_link_libraries(stereo_component
  ${INHERITED_LIBS}
  CUDA::cudart
  CUDA::nppicc
  yaml-cpp
  Threads::Threads
  vpi
  dl
  ${OpenCV_LIBS}
)

rclcpp_components_register_nodes(stereo_component "parallax::ros::StereoNode")

install(TARGETS stereo_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_package()
