#!/usr/bin/env zsh
set -eo pipefail

source /opt/ros/humble/setup.zsh

if [ -f /opt/rplidar_ws/install/setup.zsh ]; then
    source /opt/rplidar_ws/install/setup.zsh
fi

if [ -f /opt/nvblox_ws/install/setup.zsh ]; then
    source /opt/nvblox_ws/install/setup.zsh
fi

if [ -f /workspace/Parallax/ros2_ws/install/setup.zsh ]; then
    source /workspace/Parallax/ros2_ws/install/setup.zsh
fi

exec "$@"