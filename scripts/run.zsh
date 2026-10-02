#!/usr/bin/env zsh
set -eo pipefail

cd "$(dirname "$0")/.."

docker compose up -d parallax

docker compose exec parallax zsh -lc '
    set -eo pipefail
    cd /workspace/Parallax

    source /opt/ros/humble/setup.zsh
    source /opt/rplidar_ws/install/setup.zsh
    
    cd ros2_ws

    colcon build --symlink-install

    source install/setup.zsh

    exec ros2 launch bringup bringup.launch.py
'