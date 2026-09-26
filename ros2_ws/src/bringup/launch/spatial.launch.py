from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    share = FindPackageShare('bringup')
    camera_config = PathJoinSubstitution([share, 'config', 'camera.yaml'])
    isp_config = PathJoinSubstitution([share, 'config', 'isp.yaml'])
    calibration_dir = (
        '/workspace/Parallax/config/camera/calibration/results/rectification'
    )

    camera = ComposableNode(
        package='camera',
        plugin='parallax::ros::StereoNode',
        name='stereo_camera',
        parameters=[{
            'camera_config': camera_config,
            'isp_config': isp_config,
            'calibration_dir': calibration_dir,
            'preview_fps': 12,
            'jpeg_quality': 80,
            'diagnostics': True,
        }],
    )

    spatial_config = PathJoinSubstitution([
        FindPackageShare('bringup'), 'config', 'spatial.yaml'
    ])

    # Computation consumes a bounded, downscaled stereo branch. The full
    # rectified camera topics remain observation products and never depend on
    # disparity, SLAM, nvblox, Foxglove, or their queues.
    left_resize = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        name='spatial_left_resize',
        parameters=[{
            # ResizeNode resizes; it is not our RGB->mono conversion stage.
            # Supplying the real input geometry also sizes its GXF pool for
            # the actual RGB8 960x600 output instead of a mono-sized block.
            'input_width': 1920,
            'input_height': 1200,
            'output_width': 960,
            'output_height': 600,
            'keep_aspect_ratio': False,
            'disable_padding': True,
            'encoding_desired': 'rgb8',
            'input_qos': 'SENSOR_DATA',
            'output_qos': 'SENSOR_DATA',
        }],
        remappings=[
            ('image', '/compute/stereo/left/image_rect'),
            ('camera_info', '/stereo/left/camera_info'),
            ('resize/image', '/spatial/left/image_rect'),
            ('resize/camera_info', '/spatial/left/camera_info_unused'),
        ],
    )

    left_mono = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        name='spatial_left_mono',
        parameters=[{
            'image_width': 960,
            'image_height': 600,
            'encoding_desired': 'mono8',
            'input_qos': 'SENSOR_DATA',
            'output_qos': 'SENSOR_DATA',
        }],
        remappings=[
            ('image_raw', '/spatial/left/image_rect'),
            ('image', '/spatial/left/image_rect_mono'),
        ],
    )

    right_resize = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        name='spatial_right_resize',
        parameters=[{
            'input_width': 1920,
            'input_height': 1200,
            'output_width': 960,
            'output_height': 600,
            'keep_aspect_ratio': False,
            'disable_padding': True,
            'encoding_desired': 'rgb8',
            'input_qos': 'SENSOR_DATA',
            'output_qos': 'SENSOR_DATA',
        }],
        remappings=[
            ('image', '/compute/stereo/right/image_rect'),
            ('camera_info', '/stereo/right/camera_info'),
            ('resize/image', '/spatial/right/image_rect'),
            ('resize/camera_info', '/spatial/right/camera_info_unused'),
        ],
    )

    right_mono = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        name='spatial_right_mono',
        parameters=[{
            'image_width': 960,
            'image_height': 600,
            'encoding_desired': 'mono8',
            'input_qos': 'SENSOR_DATA',
            'output_qos': 'SENSOR_DATA',
        }],
        remappings=[
            ('image_raw', '/spatial/right/image_rect'),
            ('image', '/spatial/right/image_rect_mono'),
        ],
    )

    disparity = ComposableNode(
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::DisparityNode',
        name='disparity_node',
        parameters=[{
            'backend': 'CUDA',
            'max_disparity': 128.0,
            'input_qos': 'SENSOR_DATA',
            'output_qos': 'SENSOR_DATA',
        }],
        remappings=[
            ('left/image_rect', '/spatial/left/image_rect'),
            ('right/image_rect', '/spatial/right/image_rect'),
            ('left/camera_info', '/spatial/left/camera_info'),
            ('right/camera_info', '/spatial/right/camera_info'),
            ('disparity', '/stereo/disparity'),
        ],
    )

    depth = ComposableNode(
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::DisparityToDepthNode',
        name='disparity_to_depth_node',
        parameters=[{
            'input_qos': 'SENSOR_DATA',
            'output_qos': 'SENSOR_DATA',
        }],
        remappings=[
            ('disparity', '/stereo/disparity'),
            ('depth', '/stereo/depth'),
        ],
    )

    visual_slam = ComposableNode(
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        name='visual_slam',
        parameters=[{
            'num_cameras': 2,
            'rectified_images': True,
            'enable_image_denoising': False,
            'enable_imu_fusion': False,
            'enable_slam_visualization': False,
            'enable_observations_view': False,
            'enable_landmarks_view': False,
            'sync_matching_threshold_ms': 20.0,
            'image_buffer_size': 8,
            'publish_map_to_odom_tf': True,
            'publish_odom_to_base_tf': True,
            'map_frame': 'map',
            'odom_frame': 'odom',
            'base_frame': 'base_link',
        }],
        remappings=[
            ('visual_slam/image_0', '/spatial/left/image_rect_mono'),
            ('visual_slam/camera_info_0', '/spatial/left/camera_info'),
            ('visual_slam/image_1', '/spatial/right/image_rect_mono'),
            ('visual_slam/camera_info_1', '/spatial/right/camera_info'),
        ],
    )

    nvblox = ComposableNode(
        package='nvblox_ros',
        plugin='nvblox::NvbloxNode',
        name='nvblox_node',
        parameters=[spatial_config],
        remappings=[
            ('camera_0/depth/image', '/stereo/depth'),
            ('camera_0/depth/camera_info', '/spatial/left/camera_info'),
        ],
    )

    container = ComposableNodeContainer(
        name='spatial_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            camera,
            left_resize,
            right_resize,
            left_mono,
            right_mono,
            disparity,
            depth,
            visual_slam,
            nvblox,
        ],
        output='screen',
    )

    # Physical camera extrinsics are owned here. cuVSLAM owns map->odom and
    # odom->base_link; do not add competing dynamic/static world transforms.
    left_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='left_camera_tf',
        arguments=[
            '0', '0', '0',
            '-0.5', '0.5', '-0.5', '0.5',
            'base_link', 'left_camera_optical_frame',
        ],
    )

    right_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='right_camera_tf',
        arguments=[
            '0', '-0.09659880643', '0',
            '-0.5', '0.5', '-0.5', '0.5',
            'base_link', 'right_camera_optical_frame',
        ],
    )

    return LaunchDescription([left_tf, right_tf, container])
