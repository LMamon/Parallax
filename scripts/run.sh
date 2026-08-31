#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

docker compose up -d parallax

docker compose exec parallax bash -lc '
    cd /workspace/Parallax
    export PYTHONPATH=/workspace/Parallax/python:${PYTHONPATH:-}
    exec ./build/parallax
'