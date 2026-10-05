from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    share = FindPackageShare('bringup')

    spatial_config = PathJoinSubstitution([share, 'config', 'spatial.yaml'])

    topics = PathJoinSubstitution([share, 'config', 'whitelist.yaml'])
    
    nvblox = ComposableNode(
        package='nvblox_ros',
        plugin='nvblox::NvbloxNode',
        name='nvblox_node',
        parameters=[spatial_config, {
                'use_depth': False,
                'decay_tsdf_rate_hz': 0.0,
                'clear_map_outside_radius_rate_hz': 0.0,
            },],
    )

    container = ComposableNodeContainer(
        name='map_playback_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[nvblox],
        output='screen',
    )

    map_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_playback_tf',
        arguments=[
            '0', '0', '0',
            '0', '0', '0', '1',
            'map', 'base_link',
        ],
    )

    foxglove = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        parameters=[
            topics,
            {
                'send_buffer_limit': 10000000,
                'max_qos_depth': 2,
            },
        ],
    )

    return LaunchDescription([map_tf, container, foxglove])