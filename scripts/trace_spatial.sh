#!/usr/bin/env bash
set -eo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

docker compose exec parallax bash -lc '
  source /opt/ros/humble/setup.bash
  source /workspace/Parallax/ros2_ws/install/setup.bash
  exec python3 /workspace/Parallax/tools/trace_spatial_pipeline.py
'
