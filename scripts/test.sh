#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${PARALLAX_CONTAINER:-parallax}"
REPO="${PARALLAX_REPO:-/workspace/Parallax}"
JOBS="${PARALLAX_JOBS:-$(nproc)}"

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
    echo "error: container '$CONTAINER' does not exist" >&2
    echo "run: docker compose up -d $CONTAINER" >&2
    exit 1
fi

if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" != "true" ]; then
    echo "error: container '$CONTAINER' is not running" >&2
    echo "run: docker compose up -d $CONTAINER" >&2
    exit 1
fi

docker exec \
    -e PARALLAX_JOBS="$JOBS" \
    "$CONTAINER" bash -lc '
        set -euo pipefail
        cd "'"$REPO"'"

        cmake -S . -B build
        cmake --build build -j"${PARALLAX_JOBS}"
        ctest --test-dir build --output-on-failure
    '
