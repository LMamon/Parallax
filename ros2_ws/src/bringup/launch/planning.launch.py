import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    planning_config = os.path.join(
        get_package_share_directory("planning"),
        "config",
        "planning.yaml",
    )

    planner = Node(
        package="planning",
        executable="planning_node",
        name="planner",
        output="screen",
        parameters=[planning_config],
    )

    return LaunchDescription([planner])
