#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")/.."

docker compose up -d parallax

docker compose exec parallax bash -lc '
    set -eo pipefail
    cd /workspace/Parallax

    source /opt/ros/humble/setup.bash

    # The camera ROS adapter statically links root-build archives.
    # Reconfigure and incrementally rebuild them so source changes cannot
    # leave the ROS overlay linked to stale objects.
    cmake -S . -B build -DCUVSLAM_ROOT=/workspace/Parallax/.deps/cuvslam
    cmake --build build -j"$(nproc)" --target \
        parallax_core \
        parallax_cuda \
        parallax_vpi \
        parallax_camera \
        parallax_isp \
        parallax_stereo

    # Rebuild the overlay before launch so source changes cannot run through
    # a stale installed stereo node.
    cd ros2_ws
    colcon build --symlink-install
    cd ..

    source ros2_ws/install/setup.bash

    exec ros2 launch bringup bringup.launch.py
'
