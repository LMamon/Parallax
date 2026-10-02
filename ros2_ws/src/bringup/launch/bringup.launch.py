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
    topics = PathJoinSubstitution([share, 'config', 'whitelist.yaml'])
    lidar_config = PathJoinSubstitution([share, 'config', 'lidar.yaml'])
    spatial_launch = PathJoinSubstitution([share, 'launch', 'spatial.launch.py'])

    calibration_dir = (
        '/workspace/Parallax/config/camera/calibration/results/rectification'
    )

    spatial = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(spatial_launch)
    )

    # Independent 2D LiDAR branch; deliberately not connected to nvblox.
    lidar = Node(
        package='rplidar_ros',
        executable='rplidar_node',
        name='rplidar_node',
        output='screen',
        parameters=[lidar_config],
        remappings=[('scan', '/scan')],
    )

    # Original calibration is stereo_body-centered. Current base_link is at
    # the left-camera origin, 0.0482994032149552 m from stereo_body.
    lidar_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='lidar_tf',
        arguments=[
            '-0.034925', '0', '0.028575',
            '0', '0', '0', '1',
            'base_link', 'lidar',
        ],
    )

    foxglove = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        parameters=[
            topics, {
            # Foxglove is an observability boundary, not a wildcard subscriber
            # into the compute graph.
            'send_buffer_limit': 10000000,
            'max_qos_depth': 2
        }],
    )

    return LaunchDescription([spatial, lidar, lidar_tf, foxglove])
