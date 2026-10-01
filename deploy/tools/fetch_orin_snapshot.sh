#!/usr/bin/env bash
# Download the W07 snapshot of the robot's deploy tree (Orin-built binaries,
# TensorRT engines for TensorRT 8.5.2, sources as of 2026-09-24) from the
# repository's GitHub Releases, check its SHA256, and optionally extract it.
#
#   bash deploy/tools/fetch_orin_snapshot.sh            # download + verify into the current directory
#   bash deploy/tools/fetch_orin_snapshot.sh --extract  # ... then extract to ~/kdw_deploy (the path the robot guides use)
#
# The snapshot is history, not the current code: deploy/ in this repository is
# newer. On a robot, building from deploy/ (deploy/README.md, "Porting to the
# robot") is the supported path; the snapshot is for comparing against what ran
# in W07, or for an Orin where building is not possible.
set -euo pipefail
NAME=r1_orin_deploy_w07.tar.gz
URL="https://github.com/kedianwen/LDRQ/releases/download/orin-snapshot-w07/$NAME"
SHA256=691a08dd1e9b2d35506aa5819c038ccb3a752c13c3a408a901b7644fd203ae54

if [[ ! -f "$NAME" ]]; then
  if command -v curl >/dev/null; then curl -fL --retry 3 -o "$NAME.part" "$URL"
  else wget -O "$NAME.part" "$URL"; fi
  mv "$NAME.part" "$NAME"
fi
echo "$SHA256  $NAME" | sha256sum -c -
if [[ "${1:-}" == "--extract" ]]; then
  tar xzf "$NAME" -C "$HOME"      # tar.gz keeps the execute bits; a zip would not
  echo "extracted to $HOME/kdw_deploy"
fi
