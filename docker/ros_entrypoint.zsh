#!/usr/bin/env zsh
set -eo pipefail

source /opt/ros/humble/setup.zsh

if [ -f /workspace/Parallax/ros2_ws/install/setup.zsh ]; then
    source /workspace/Parallax/ros2_ws/install/setup.zsh
fi

exec "$@"