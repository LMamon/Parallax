from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    planner = Node(package='planning',
                   executable='planning_node',
                   name='planner',
                   output='screen')

    return LaunchDescription([planner])