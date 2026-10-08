cmake_minimum_required(VERSION 3.18)
project(visualization LANGUAGES C CXX)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Color.msg"
  "msg/Point2.msg"
  "msg/KeyValuePair.msg"
  "msg/PointsAnnotation.msg"
  "msg/TextAnnotation.msg"
  "msg/ImageAnnotations.msg"
  DEPENDENCIES builtin_interfaces
)

add_executable(perception_visualizer
  src/perception_visualizer.cpp
)
target_compile_features(perception_visualizer PRIVATE cxx_std_17)
ament_target_dependencies(perception_visualizer
  rclcpp
  interfaces
)

rosidl_get_typesupport_target(
  visualization_typesupport
  ${PROJECT_NAME}
  "rosidl_typesupport_cpp"
)
target_link_libraries(perception_visualizer "${visualization_typesupport}")

install(TARGETS perception_visualizer
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
