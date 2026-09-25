#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /workspace/Parallax/ros2_ws/install/setup.bash

echo "=== required nodes ==="
ros2 node list | grep -E \
  '^/(disparity_node|disparity_to_depth_node|foxglove_bridge|nvblox_node|spatial_left_resize|spatial_right_resize|stereo_camera|visual_slam)$' \
  | sort || true

echo
echo "=== computational image boundary ==="
ros2 topic info /spatial/left/image_rect || true
ros2 topic info /spatial/right/image_rect || true
ros2 topic echo /spatial/left/camera_info --once --field width || true
ros2 topic echo /spatial/left/camera_info --once --field height || true

echo
echo "=== spatial topics ==="
ros2 topic list | grep -E \
  '^/(spatial|stereo/(depth|disparity)|visual_slam|nvblox_node)' | sort || true

echo
echo "=== depth sample ==="
timeout 5 ros2 topic echo /stereo/depth --once --field header || true

echo
echo "=== cuVSLAM odometry sample ==="
timeout 5 ros2 topic echo /visual_slam/tracking/odometry --once --field header || true

echo
echo "=== TF: map -> base_link ==="
timeout 8 ros2 run tf2_ros tf2_echo map base_link || true

echo
echo "=== TF: base_link -> left optical ==="
timeout 5 ros2 run tf2_ros tf2_echo base_link left_camera_optical_frame || true

echo
echo "=== nvblox services ==="
ros2 service list | grep '^/nvblox_node/' | sort || true
