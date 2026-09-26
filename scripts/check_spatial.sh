#!/usr/bin/env bash
set -eo pipefail

ROOT="${ROOT:-$(git rev-parse --show-toplevel)}"
cd "$ROOT"

if ! docker compose ps --status running --services | grep -qx parallax; then
  echo "ERROR: parallax container is not running." >&2
  exit 1
fi

docker compose exec -T parallax bash -lc '
set -eo pipefail
source /opt/ros/humble/setup.bash
source /workspace/Parallax/ros2_ws/install/setup.bash

fail=0

node_list="$(ros2 node list)"

require_node() {
  if grep -Fxq "$1" <<<"$node_list"; then
    printf "PASS node %s\n" "$1"
  else
    printf "FAIL node %s\n" "$1"
    fail=1
  fi
}

echo "=== required nodes ==="
for node in /stereo_camera /spatial_left_resize /spatial_right_resize /spatial_left_mono /spatial_right_mono /disparity_node /disparity_to_depth_node /visual_slam /nvblox_node /foxglove_bridge; do
  require_node "$node"
done

echo
echo "=== spatial boundary ==="
width="$(timeout 5 ros2 topic echo /spatial/left/camera_info --qos-reliability best_effort --once --field width 2>/dev/null | grep -Eo "[0-9]+" | head -1 || true)"
height="$(timeout 5 ros2 topic echo /spatial/left/camera_info --qos-reliability best_effort --once --field height 2>/dev/null | grep -Eo "[0-9]+" | head -1 || true)"
if [[ "$width" == "960" && "$height" == "600" ]]; then
  echo "PASS camera_info 960x600"
else
  echo "FAIL camera_info ${width:-?}x${height:-?}"
  fail=1
fi

echo
echo "=== depth ==="
if timeout 8 ros2 topic echo /stereo/depth --qos-reliability best_effort --once --field header >/tmp/parallax_depth_check 2>&1; then
  echo "PASS /stereo/depth"
  cat /tmp/parallax_depth_check
else
  echo "FAIL /stereo/depth"
  cat /tmp/parallax_depth_check || true
  fail=1
fi
rm -f /tmp/parallax_depth_check

echo
echo "=== cuVSLAM odometry ==="
if timeout 8 ros2 topic echo /visual_slam/tracking/odometry --once --field header >/tmp/parallax_odom_check 2>&1; then
  echo "PASS /visual_slam/tracking/odometry"
  cat /tmp/parallax_odom_check
else
  echo "FAIL /visual_slam/tracking/odometry"
  cat /tmp/parallax_odom_check || true
  fail=1
fi
rm -f /tmp/parallax_odom_check

echo
echo "=== TF map -> base_link ==="
timeout 8 ros2 run tf2_ros tf2_echo map base_link >/tmp/parallax_tf_check 2>&1 || true
if grep -q "Translation:" /tmp/parallax_tf_check; then
  echo "PASS map -> base_link"
  tail -n 12 /tmp/parallax_tf_check
else
  echo "FAIL map -> base_link"
  cat /tmp/parallax_tf_check || true
  fail=1
fi
rm -f /tmp/parallax_tf_check

echo
echo "=== nvblox ESDF interface ==="
service_list="$(ros2 service list)"
if grep -Fxq /nvblox_node/get_esdf_and_gradient <<<"$service_list"; then
  echo "PASS /nvblox_node/get_esdf_and_gradient"
else
  echo "FAIL /nvblox_node/get_esdf_and_gradient"
  fail=1
fi

echo
echo "=== producer rates ==="
echo "See camera_diag in T1 for source/compute rates."
echo "Use nvblox shutdown statistics for integrated depth rate."

exit "$fail"
'
