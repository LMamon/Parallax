#!/bin/bash
set -eo pipefail

source /opt/ros/humble/setup.bash

if [ -f /workspace/Parallax/ros2_ws/install/setup.bash ]; then
    source /workspace/Parallax/ros2_ws/install/setup.bash
fi

exec "$@"