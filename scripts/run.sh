#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")/.."

docker compose up -d parallax

docker compose exec parallax bash -lc '
    set -eo pipefail
    cd /workspace/Parallax

    source /opt/ros/humble/setup.bash

    if [ ! -f ros2_ws/install/setup.bash ]; then
        cd ros2_ws
        colcon build --symlink-install
        cd ..
    fi

    source ros2_ws/install/setup.bash

    exec ros2 launch bringup bringup.launch.py
'
