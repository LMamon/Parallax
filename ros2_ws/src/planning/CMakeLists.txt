cmake_minimum_required(VERSION 3.18)
project(planning)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(nvblox_msgs REQUIRED)
find_package(ompl REQUIRED)

add_library(planning_core
  src/esdf_grid.cpp
  src/esdf_motion_validator.cpp
)

target_compile_features(planning_core PUBLIC cxx_std_17)

target_include_directories(planning_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OMPL_INCLUDE_DIRS}
)

ament_target_dependencies(planning_core
  nvblox_msgs
)

target_link_libraries(planning_core
  ${OMPL_LIBRARIES}
)

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
  planning_core
  ${OMPL_LIBRARIES}
)

install(
  TARGETS
    planning_core
    planning_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()
