cmake_minimum_required(VERSION 3.18)
project(interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME} 
    "msg/Detection2D.msg" 
    "msg/DetectionSet.msg" 
    "msg/Track2D.msg" 
    "msg/PerceptionState.msg" 
    "srv/SetPrompt.srv" 
    "srv/ClearPrompt.srv" 
    "srv/StartTracking.srv" 
    "srv/StopTracking.srv" 

    DEPENDENCIES 
        std_msgs 
        geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
