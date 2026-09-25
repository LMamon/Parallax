#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")/.."

docker compose exec parallax bash -lc '
    set -eo pipefail
    source /opt/ros/humble/setup.bash
    source /workspace/Parallax/ros2_ws/install/setup.bash

    echo "=== stereo topics ==="
    ros2 topic list | grep "^/stereo/" | sort

    echo
    echo "=== left CameraInfo ==="
    timeout 5 ros2 topic echo --once /stereo/left/camera_info || true

    echo
    echo "=== right CameraInfo ==="
    timeout 5 ros2 topic echo --once /stereo/right/camera_info || true

    echo
    echo "=== left mono rate ==="
    timeout 7 ros2 topic hz /stereo/left/image_rect || true

    echo
    echo "=== right mono rate ==="
    timeout 7 ros2 topic hz /stereo/right/image_rect || true

    echo
    echo "=== compressed color preview rate ==="
    timeout 7 ros2 topic hz /stereo/left/image_rect_color/compressed || true

    echo
    echo "=== compressed color preview bandwidth ==="
    timeout 5 ros2 topic bw /stereo/left/image_rect_color/compressed || true
'
