# Parallax v3.1 References

Reference links for the Parallax v3.1 integration plan. Keep this file
separate from the implementation/orchestration plan so the latter can
stay focused on execution.

## Existing Parallax / RXSIM Reference

-   **RXSIM ROS/Gazebo source tree**\
    https://github.com/LMamon/RXSIM/tree/main/rosgz/src

## ROS 2 Camera Ingress and Calibration

-   **v4l2_camera --- ROS 2 V4L2 camera driver (upstream source)**\
    https://gitlab.com/boldhearts/ros2_v4l2_camera

-   **v4l2_camera --- ROS Index**\
    https://index.ros.org/p/v4l2_camera/

-   **ROS image_common --- image_transport, camera calibration parsers,
    camera_info_manager**\
    https://github.com/ros-perception/image_common

-   **camera_info_manager --- ROS Index**\
    https://index.ros.org/p/camera_info_manager/

The existing Parallax stereo calibration metadata should be integrated
through standard `sensor_msgs/CameraInfo` / `camera_info_manager`
mechanisms rather than defining another calibration representation.

## NVIDIA Isaac ROS 3.2

-   **Isaac ROS NITROS**\
    https://nvidia-isaac-ros.github.io/v/release-3.2/repositories_and_packages/isaac_ros_nitros/index.html

-   **Isaac ROS Image Pipeline**\
    https://nvidia-isaac-ros.github.io/v/release-3.2/repositories_and_packages/isaac_ros_image_pipeline/index.html

-   **Isaac ROS Image Pipeline --- NITROS Acceleration**\
    https://nvidia-isaac-ros.github.io/v/release-3.2/repositories_and_packages/isaac_ros_image_pipeline/index.html#isaac-ros-nitros-acceleration

-   **Isaac ROS Stereo Image Proc --- Disparity / Depth / Point Cloud**\
    https://nvidia-isaac-ros.github.io/v/release-3.2/repositories_and_packages/isaac_ros_image_pipeline/isaac_ros_stereo_image_proc/index.html

-   **Isaac ROS Visual SLAM / cuVSLAM**\
    https://nvidia-isaac-ros.github.io/v/release-3.2/repositories_and_packages/isaac_ros_visual_slam/index.html

-   **Isaac ROS nvblox**\
    https://nvidia-isaac-ros.github.io/v/release-3.2/repositories_and_packages/isaac_ros_nvblox/index.html

## Planning

-   **OMPL**\
    https://github.com/ompl/ompl

-   **OMPL documentation**\
    https://ompl.kavrakilab.org/

## RPLIDAR C1

-   **SLAMTEC sllidar_ros2 --- ROS 2 driver**\
    https://github.com/Slamtec/sllidar_ros2

-   **SLAMTEC rplidar_ros --- older ROS package/reference**\
    https://github.com/Slamtec/rplidar_ros

For v3.1, prefer `sllidar_ros2`; its upstream package explicitly
supports ROS 2 and the RPLIDAR C1.

## Foxglove / ROS 2

-   **Foxglove ROS 2 setup**\
    https://docs.foxglove.dev/docs/getting-started/frameworks/ros2

-   **Foxglove Bridge documentation**\
    https://docs.foxglove.dev/docs/fleet/bridge

-   **foxglove_bridge ROS 2 source**\
    https://github.com/foxglove/foxglove-sdk/tree/main/ros/src/foxglove_bridge

## ROS 2 Runtime / Lifecycle / Composition

-   **ROS 2 Managed Nodes / Lifecycle design**\
    https://design.ros2.org/articles/node_lifecycle.html

-   **ROS 2 Humble lifecycle CLI documentation**\
    https://docs.ros.org/en/humble/p/ros2lifecycle/

-   **ROS 2 Composition**\
    https://docs.ros.org/en/humble/Concepts/Intermediate/About-Composition.html

These are relevant to the v3.1 capability/resource model: use existing
ROS lifecycle and composition mechanisms where they fit before adding
custom orchestration.

------------------------------------------------------------------------

## Milestone Reference Map

### Milestone 1 --- Physical stereo ingress

Relevant references:

-   `v4l2_camera`
-   `image_common` / `camera_info_manager`
-   Isaac ROS Image Pipeline
-   Isaac ROS NITROS

Target:

``` text
AR0234 SBS
  -> v4l2_camera
  -> left/right CropNode
  -> CameraInfo
  -> RectifyNode
  -> clean rectified stereo in Foxglove
```

### Milestone 2 --- Stereo disparity and depth

Relevant references:

-   Isaac ROS Stereo Image Proc
-   Isaac ROS Image Pipeline
-   NITROS
-   Foxglove Bridge

Target:

``` text
rectified stereo
  -> DisparityNode
  -> DisparityToDepthNode
  -> depth / optional point cloud
```

### Milestone 3 --- Visual localization

Relevant references:

-   RXSIM
-   Isaac ROS Visual SLAM

Target:

``` text
rectified stereo
  -> cuVSLAM
  -> pose / odometry / TF
```

### Milestone 4 --- Persistent 3D mapping

Relevant references:

-   RXSIM
-   Isaac ROS nvblox

Target:

``` text
depth + TF
  -> nvblox
  -> persistent TSDF
  -> 3D ESDF
  -> Foxglove
```

### Milestone 5 --- 3D planning

Relevant references:

-   Isaac ROS nvblox
-   OMPL
-   ROS 2 TF / standard messages
-   Foxglove

Target:

``` text
manual goal
  -> current TF pose
  -> nvblox ESDF snapshot
  -> EsdfCache
  -> OMPL R^3 / RRTConnect
  -> nav_msgs/Path
  -> Foxglove
```

### Milestone 6 --- Semantic goal

Relevant references:

-   existing Parallax NanoOWL / EfficientViT-SAM / Object3D code
-   OMPL
-   nvblox

Target:

``` text
NanoOWL + SAM
  -> Object3D / tracked target
  -> approach point
  -> existing planning pipeline
```

------------------------------------------------------------------------

## Initial Stereo Composition Sketch

This is a working integration sketch, not an API contract. Confirm exact
Isaac ROS 3.2 component names, parameters, topic names, and supported
formats against the release-3.2 documentation while implementing.

``` python
processing_container = ComposableNodeContainer(
    name='parallax_v3_container',
    namespace='',
    package='rclcpp_components',
    executable='component_container_mt',
    composable_node_descriptions=[
        ComposableNode(
            package='v4l2_camera',
            plugin='v4l2_camera::V4L2CameraNode',
            name='v4l2_camera_ingress',
            parameters=[{
                'image_width': 3840,
                'image_height': 1200,
            }],
            remappings=[
                ('image_raw', '/stereo/stitched_image'),
            ],
        ),

        ComposableNode(
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::CropNode',
            name='crop_left_gpu',
            parameters=[{
                'input_width': 3840,
                'input_height': 1200,
                'crop_width': 1920,
                'crop_height': 1200,
                'x_offset': 0,
                'y_offset': 0,
            }],
            remappings=[
                ('image', '/stereo/stitched_image'),
                ('crop/image', '/left/image'),
            ],
        ),

        ComposableNode(
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::CropNode',
            name='crop_right_gpu',
            parameters=[{
                'input_width': 3840,
                'input_height': 1200,
                'crop_width': 1920,
                'crop_height': 1200,
                'x_offset': 1920,
                'y_offset': 0,
            }],
            remappings=[
                ('image', '/stereo/stitched_image'),
                ('crop/image', '/right/image'),
            ],
        ),

        # CameraInfo + RectifyNode
        # DisparityNode + DisparityToDepthNode
        # cuVSLAM
        # nvblox
    ],
    output='screen',
)
```

## Notes

-   Pin NVIDIA references to **Isaac ROS release 3.2** for this
    migration instead of silently using newer documentation.
-   Use the actual RXSIM launch/config files as the reference for
    already-solved cuVSLAM/nvblox wiring.
-   Prefer standard ROS 2 interfaces and lifecycle behavior over
    recreating Parallax-specific infrastructure.
-   Treat the stereo composition above as a starting sketch. Validate
    exact component/plugin names and parameter schemas against the
    installed Humble / Isaac ROS 3.2 packages before implementation.
