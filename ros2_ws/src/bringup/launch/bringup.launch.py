from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    camera_config = PathJoinSubstitution([
        FindPackageShare('bringup'), 'config', 'camera.yaml'
    ])
    isp_config = PathJoinSubstitution([
        FindPackageShare('bringup'), 'config', 'isp.yaml'
    ])

    # Calibration remains the inherited generated artifact for this gate.
    calibration_dir = (
        '/workspace/Parallax/config/camera/calibration/results/rectification'
    )

    return LaunchDescription([
        Node(
            package='camera',
            executable='stereo_node',
            name='stereo_camera',
            output='screen',
            parameters=[{
                'camera_config': camera_config,
                'isp_config': isp_config,
                'calibration_dir': calibration_dir,
                'preview_fps': 20,
                'jpeg_quality': 85,
            }],
        ),
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            output='screen',
        ),
    ])
