#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")/.."

docker compose up -d parallax

docker compose exec parallax bash -lc '
    set -eo pipefail
    cd /workspace/Parallax

    source /opt/ros/humble/setup.bash

    # The camera ROS adapter still links the proven sensor-specific ingress,
    # ISP and calibrated VPI remap from the inherited build.
    if [ ! -f build/src/stereo/libparallax_stereo.a ]; then
        cmake -S . -B build -DCUVSLAM_ROOT=/workspace/Parallax/.deps/cuvslam
        cmake --build build -j"$(nproc)"
    fi

    # Rebuild the overlay before launch so source changes cannot run through
    # a stale installed stereo node.
    cd ros2_ws
    colcon build --symlink-install
    cd ..

    source ros2_ws/install/setup.bash

    exec ros2 launch bringup bringup.launch.py
'
