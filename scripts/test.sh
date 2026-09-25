#!/usr/bin/env bash
set -eo pipefail

CONTAINER="${PARALLAX_CONTAINER:-parallax}"
REPO="${PARALLAX_REPO:-/workspace/Parallax}"
JOBS="${PARALLAX_JOBS:-$(nproc)}"

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
    echo "error: container '$CONTAINER' does not exist" >&2
    echo "run: docker compose up -d $CONTAINER" >&2
    exit 1
fi

if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" != "true" ]; then
    echo "error: container '$CONTAINER' is not running" >&2
    echo "run: docker compose up -d $CONTAINER" >&2
    exit 1
fi

docker exec \
    -e PARALLAX_JOBS="$JOBS" \
    "$CONTAINER" bash -lc '
        set -eo pipefail
        cd "'"$REPO"'"

        # Preserve the inherited test gate until ROS replacements have passed
        # their own hardware gates and the old implementation can be retired.
        cmake -S . -B build -DCUVSLAM_ROOT=/workspace/Parallax/.deps/cuvslam
        cmake --build build -j"${PARALLAX_JOBS}"
        ctest --test-dir build --output-on-failure

        # The v3.1 overlay is now a second hard gate, not a separate workflow.
        source /opt/ros/humble/setup.bash
        cd ros2_ws
        colcon build --symlink-install
        source install/setup.bash

        ros2 pkg prefix bringup >/dev/null
        ros2 pkg prefix camera >/dev/null
        ros2 pkg prefix isaac_ros_visual_slam >/dev/null
        ros2 pkg prefix isaac_ros_nvblox >/dev/null
        ros2 pkg prefix isaac_ros_stereo_image_proc >/dev/null
        ros2 pkg prefix foxglove_bridge >/dev/null
    '
