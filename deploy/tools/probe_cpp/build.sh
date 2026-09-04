#!/usr/bin/env bash
# Build the read-only LowState probe against the system unitree_sdk2.
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
else
    # Fallback: no cmake, no package config, just the install layout.
    echo "cmake not found, compiling directly"
    g++ -std=c++17 -O2 -o probe_lowstate probe_lowstate.cpp \
        -I/usr/local/include \
        -L/usr/local/lib -lunitree_sdk2 -lddscxx -lddsc -lpthread
    echo "ok -> $PWD/probe_lowstate"
fi
