from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Bring up the ROS-facing shell before hardware processing is added."""
    return LaunchDescription([
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            output='screen',
        ),
    ])
