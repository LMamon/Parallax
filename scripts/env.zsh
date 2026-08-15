#!/usr/bin/env zsh

export PARALLAX_ROOT="${PARALLAX_ROOT:-${${(%):-%N}:A:h:h}}"
export PARALLAX_WS="$PARALLAX_ROOT/ros_ws"

export ROS_DOMAIN_ID=2
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

load_parallax() {
    source /opt/ros/humble/setup.zsh

    if [ -f "$PARALLAX_WS/install/setup.zsh" ]; then
        source "$PARALLAX_WS/install/setup.zsh"
    else
        echo "Parallax workspace not built."
    fi
}

load_parallax