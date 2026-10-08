cmake_minimum_required(VERSION 3.18)
project(planning)

find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(nvblox_msgs REQUIRED)
find_package(ompl REQUIRED)

add_executable(planning_node
  src/planning_node.cpp
)

target_compile_features(planning_node PRIVATE cxx_std_17)

target_include_directories(planning_node
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OMPL_INCLUDE_DIRS}
)

ament_target_dependencies(planning_node
    rclcpp
    geometry_msgs
    nav_msgs
    tf2
    tf2_ros
    tf2_geometry_msgs
    nvblox_msgs
)

target_link_libraries(planning_node
  ${OMPL_LIBRARIES}
)

install(
  TARGETS planning_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()