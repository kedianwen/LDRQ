#!/usr/bin/env bash
# Dev box: build the robot's LLM runtime package -- Ollama for JetPack 5 plus the
# models -- as one tar.gz that extracts to ./llm/ollama and ./llm/models under
# ~/kdw_deploy, next to the scripts from the stage C code package.
#
#   bash llm/make_runtime.sh
#   MODELS="qwen3:1.7b" OUT=~/r1_llm_runtime_small.tar.gz bash llm/make_runtime.sh
#
# Inputs (see llm/README.md for how they were fetched and checked):
#   $DL     the release tarballs: ollama-linux-arm64.tar.zst and
#           ollama-linux-arm64-jetpack5.tar.zst, with SHA256SUMS from the GitHub API
#   $STORE  a model store (OLLAMA_MODELS) with $MODELS pulled into it
#
# The robot's tar (1.30, Ubuntu 20.04) cannot read zstd, so everything is unpacked
# here and repacked as gzip. The arm64 package's cuda_v12/ and cuda_v13/ (2.2 GB)
# are for SBSA server GPUs, not Jetson, and are left out; the jetpack5 package
# brings its own CUDA 11 runner and cuBLAS. Nothing is compiled.
set -euo pipefail
VER="${VER:-v0.34.4}"
DL="${DL:-$HOME/R1process/outputs/llm/downloads}"
STORE="${STORE:-$HOME/R1process/outputs/llm/ollama_models}"
MODELS="${MODELS:-qwen3:1.7b qwen3:0.6b qwen2.5:1.5b}"
OUT="${OUT:-$HOME/r1_stageC_llm_runtime_$(date +%Y-%m-%d).tar.gz}"
GZ="$(command -v pigz || command -v gzip)"

cd "$DL"
sha256sum -c SHA256SUMS --ignore-missing
for f in ollama-linux-arm64.tar.zst ollama-linux-arm64-jetpack5.tar.zst; do
    grep -q " $f\$" SHA256SUMS || { echo "no checksum for $f in $DL/SHA256SUMS"; exit 1; }
done

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/llm/ollama" "$WORK/llm/models"
echo "unpacking the arm64 base (without cuda_v12/, cuda_v13/)..."
unzstd -c ollama-linux-arm64.tar.zst | tar -x -C "$WORK/llm/ollama" \
    --exclude='lib/ollama/cuda_v12' --exclude='lib/ollama/cuda_v13'
echo "unpacking the jetpack5 runner..."
unzstd -c ollama-linux-arm64-jetpack5.tar.zst | tar -x -C "$WORK/llm/ollama"

echo "copying models: $MODELS"
for m in $MODELS; do
    name="${m%%:*}"; tag="${m#*:}"
    [ "$tag" = "$m" ] && tag=latest
    man="manifests/registry.ollama.ai/library/$name/$tag"
    [ -f "$STORE/$man" ] || { echo "model $m is not in $STORE (ollama pull it first)"; exit 1; }
    mkdir -p "$WORK/llm/models/$(dirname "$man")"
    cp "$STORE/$man" "$WORK/llm/models/$man"
    mkdir -p "$WORK/llm/models/blobs"
    python3 - "$STORE/$man" <<'PY' | while read -r d; do cp -n "$STORE/blobs/$d" "$WORK/llm/models/blobs/$d"; done
import json, sys
m = json.load(open(sys.argv[1]))
for layer in [m["config"]] + m["layers"]:
    print(layer["digest"].replace("sha256:", "sha256-"))
PY
done

cd "$WORK/llm"
{
    echo "R1 stage C LLM runtime, built $(date -Iseconds) on $(hostname)"
    echo "Ollama $VER: ollama-linux-arm64.tar.zst + ollama-linux-arm64-jetpack5.tar.zst"
    grep -E "arm64(-jetpack5)?\.tar\.zst" "$DL/SHA256SUMS" | sed 's/^/  sha256 /'
    echo "left out: lib/ollama/cuda_v12, lib/ollama/cuda_v13 (SBSA server GPUs, not Jetson)"
    echo "models:"
    for m in $MODELS; do echo "  $m"; done
    echo "every file's sha256 is in RUNTIME_SHA256SUMS: cd ~/kdw_deploy/llm && sha256sum -c --quiet RUNTIME_SHA256SUMS"
} > RUNTIME_MANIFEST.txt
find ollama models -type f -print0 | sort -z | xargs -0 sha256sum > RUNTIME_SHA256SUMS
cd "$WORK"
echo "packing $OUT ..."
tar -c --owner=0 --group=0 -f - llm | "$GZ" > "$OUT"
ls -la "$OUT"
md5sum "$OUT"
cat "$WORK/llm/RUNTIME_MANIFEST.txt"
du -sh "$WORK/llm/ollama" "$WORK/llm/models"
