cmake_minimum_required(VERSION 3.18)
project(camera LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(vpi 3.2 REQUIRED)

set(PARALLAX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(PARALLAX_BUILD "${PARALLAX_ROOT}/build")

# Temporary compatibility seam: sensor-specific capture, ISP and calibrated
# remap stay inherited until the ROS/Isaac replacement passes hardware gates.
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

add_executable(stereo_node src/stereo_node.cpp)
target_compile_features(stereo_node PRIVATE cxx_std_17)
target_include_directories(stereo_node PRIVATE "${PARALLAX_ROOT}/include")
ament_target_dependencies(stereo_node rclcpp sensor_msgs)

# Humble's ament_target_dependencies uses the plain link signature.
target_link_libraries(stereo_node
  ${INHERITED_LIBS}
  CUDA::cudart
  CUDA::nppicc
  yaml-cpp
  Threads::Threads
  vpi
  dl
)

target_link_libraries(stereo_node ${OpenCV_LIBS})

install(TARGETS stereo_node DESTINATION lib/${PROJECT_NAME})
ament_package()
