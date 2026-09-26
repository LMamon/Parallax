from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    share = FindPackageShare('bringup')

    camera_config = PathJoinSubstitution([share, 'config', 'camera.yaml'])
    isp_config = PathJoinSubstitution([share, 'config', 'isp.yaml'])
    spatial_launch = PathJoinSubstitution([share, 'launch', 'spatial.launch.py'])

    calibration_dir = (
        '/workspace/Parallax/config/camera/calibration/results/rectification'
    )

    spatial = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(spatial_launch)
    )

    foxglove = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        parameters=[{
            # Foxglove observes ROS products; it does not own the spatial path.
            'send_buffer_limit': 10000000,
        }],
    )

    return LaunchDescription([spatial, foxglove])
