#!/usr/bin/env bash
# Build the read-only probes against the system unitree_sdk2:
#   probe_lowstate  slot mapping / IMU conventions
#   probe_lowcmd    is anything ELSE writing rt/lowcmd (the second-writer check)
#
# Deliberately NOT part of the colcon workspace: the probe must run before any
# ROS 2 environment is trusted, and it is the one thing W06 cannot start
# without. Keeping it standalone means a broken ROS setup cannot block it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if command -v cmake >/dev/null 2>&1; then
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j"$(nproc)"
    echo "ok -> $PWD/build/probe_lowstate"
    echo "ok -> $PWD/build/probe_lowcmd"
else
    # Fallback: no cmake, no package config, just the install layout.
    echo "cmake not found, compiling directly"
    for t in probe_lowstate probe_lowcmd; do
        g++ -std=c++17 -O2 -o "$t" "$t.cpp" \
            -I/usr/local/include \
            -L/usr/local/lib -lunitree_sdk2 -lddscxx -lddsc -lpthread
        echo "ok -> $PWD/$t"
    done
fi
