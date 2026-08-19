#!/usr/bin/env bash
# Provision the deployment toolchain for the R1 policy runner.
#
# Idempotent: re-running it is how you verify the workspace is still intact.
# Nothing here needs sudo -- everything lands under deploy/ so the training
# environment (conda env_isaaclab) is never touched.
#
#   ./setup.sh            # provision / repair
#   ./setup.sh --check    # verify only, change nothing
#
# On a Jetson (JetPack) you do NOT run this: TensorRT, CUDA and their headers
# come from JetPack under /usr, and env.sh detects that automatically.

set -euo pipefail

DEPLOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${DEPLOY_ROOT}/.venv"
TP="${DEPLOY_ROOT}/third_party"

# TensorRT 10.7.0, chosen by measurement rather than preference.
#
# The obvious pick was 10.3.0, the version JetPack 6.1/6.2 ships for Jetson
# Orin -- matching the target avoids API drift that only shows up on the robot.
# It does not work here: on this box (Turing sm_75, driver 580.173) TensorRT
# 10.3.0 builds an engine that runs at full speed and returns the WRONG
# numbers. Reproduced from the ONNX parser and from a hand-built network, at
# every builder optimisation level, in FP32 and FP16, from both C++ and Python
# -- while onnxruntime and PyTorch agree with each other. Any subgraph with two
# ELU layers is correct; three is wrong. 10.7.0 matches PyTorch to 5e-6.
#
# Consequence for deployment: the dev box and the robot are now on different
# TensorRT versions, so `r1_parity_check` MUST be run again on the robot after
# building the engine there. See deploy/README.md.
TRT_VERSION="10.7.0"
TRT_BRANCH="release/10.7"          # NVIDIA/TensorRT branch holding the matching public headers
ONNX_PARSER_TAG="release/10.7-GA"  # onnx/onnx-tensorrt tag holding NvOnnxParser.h

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

say()  { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m    ! %s\033[0m\n' "$*"; }
die()  { printf '\033[31m    x %s\033[0m\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------
# 0. Host prerequisites we cannot install ourselves
# --------------------------------------------------------------------------
say "checking host prerequisites"
[[ -d /opt/ros/humble ]] || die "ROS 2 humble not found at /opt/ros/humble"
command -v colcon >/dev/null || die "colcon not on PATH"
command -v cmake  >/dev/null || die "cmake not on PATH"
command -v g++    >/dev/null || die "g++ not on PATH"
[[ -x /usr/bin/python3.10 ]] || die "python3.10 not found (ROS 2 humble needs it)"
command -v nvidia-smi >/dev/null || warn "nvidia-smi missing -- engine build will fail"
printf '    ros=humble  cmake=%s  g++=%s  python=%s\n' \
    "$(cmake --version | head -1 | awk '{print $3}')" \
    "$(g++ -dumpversion)" \
    "$(/usr/bin/python3.10 --version | awk '{print $2}')"

# --------------------------------------------------------------------------
# 1. Python venv -- deliberately built on the SYSTEM python3.10, not conda.
#    ROS 2 humble is compiled against system python3.10; a conda interpreter
#    here makes colcon resolve the wrong Python and fail in confusing ways.
# --------------------------------------------------------------------------
if [[ ! -x "${VENV}/bin/python" ]]; then
    [[ ${CHECK_ONLY} -eq 1 ]] && die "venv missing"
    say "creating venv on system python3.10"
    /usr/bin/python3.10 -m venv "${VENV}"
    "${VENV}/bin/python" -m pip install -q --upgrade pip setuptools wheel
fi

if ! "${VENV}/bin/python" -c "import tensorrt" >/dev/null 2>&1; then
    [[ ${CHECK_ONLY} -eq 1 ]] && die "tensorrt not installed in venv"
    say "installing tensorrt==${TRT_VERSION} (+ onnx, numpy)"
    "${VENV}/bin/python" -m pip install -q "tensorrt==${TRT_VERSION}" "numpy<2" onnx
fi
say "python side: TensorRT $("${VENV}/bin/python" -c 'import tensorrt;print(tensorrt.__version__)')"

SP="$("${VENV}/bin/python" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"

# --------------------------------------------------------------------------
# 2. TensorRT C++ headers.
#    The pip wheel ships the shared objects but NOT the headers. NVIDIA
#    publishes the same public headers in the open-source TensorRT repo, so we
#    take them from the branch matching the installed binary and then assert
#    the version macros agree -- a header/binary mismatch here would surface as
#    silent ABI corruption, not a compile error.
# --------------------------------------------------------------------------
TRT_INC="${TP}/tensorrt/include"
if [[ ! -f "${TRT_INC}/NvInfer.h" ]]; then
    [[ ${CHECK_ONLY} -eq 1 ]] && die "TensorRT headers missing"
    say "fetching TensorRT ${TRT_VERSION} public headers"
    mkdir -p "${TRT_INC}"
    # Enumerate the directory rather than hard-coding filenames: the header set
    # changes between minor versions (10.7 dropped NvInferConsistency*.h and
    # added NvInferPluginBase.h), and a stale list fails as a bare 404.
    headers="$(curl -sSfL "https://api.github.com/repos/NVIDIA/TensorRT/contents/include?ref=${TRT_BRANCH}" \
        | "${VENV}/bin/python" -c 'import json,sys; print("\n".join(f["name"] for f in json.load(sys.stdin) if f["name"].endswith(".h")))')"
    [[ -n "${headers}" ]] || die "could not list headers for ${TRT_BRANCH}"
    base="https://raw.githubusercontent.com/NVIDIA/TensorRT/${TRT_BRANCH}/include"
    while read -r h; do
        [[ -n "${h}" ]] && curl -sSfL -o "${TRT_INC}/${h}" "${base}/${h}"
    done <<< "${headers}"
    # NvOnnxParser.h lives in the onnx-tensorrt submodule, not in include/.
    curl -sSfL -o "${TRT_INC}/NvOnnxParser.h" \
        "https://raw.githubusercontent.com/onnx/onnx-tensorrt/${ONNX_PARSER_TAG}/NvOnnxParser.h"
fi

hdr_ver="$(awk '/NV_TENSORRT_MAJOR/{M=$3} /NV_TENSORRT_MINOR/{m=$3} /NV_TENSORRT_PATCH/{p=$3} END{print M"."m"."p}' \
           "${TRT_INC}/NvInferVersion.h")"
[[ "${hdr_ver}" == "${TRT_VERSION}" ]] \
    || die "header version ${hdr_ver} != installed binary ${TRT_VERSION}"
say "headers ${hdr_ver} match the installed runtime"

# --------------------------------------------------------------------------
# 3. Link farms.
#    The wheels ship versioned sonames only (libnvinfer.so.10, libcudart.so.12);
#    the linker wants an unversioned libfoo.so. Stage symlinks rather than
#    copies so a pip upgrade is picked up without re-running this step.
# --------------------------------------------------------------------------
say "staging link farms"
mkdir -p "${TP}/tensorrt/lib" "${TP}/cuda/lib"
for so in libnvinfer.so.10 libnvonnxparser.so.10 libnvinfer_plugin.so.10; do
    [[ -f "${SP}/tensorrt_libs/${so}" ]] || die "missing ${so} in ${SP}/tensorrt_libs"
    ln -sfn "${SP}/tensorrt_libs/${so}" "${TP}/tensorrt/lib/${so}"
    ln -sfn "${SP}/tensorrt_libs/${so}" "${TP}/tensorrt/lib/${so%.10}"
done

# Not a link-time dependency: the builder dlopen()s this by exact filename when
# createInferBuilder runs, so it has to be reachable through LD_LIBRARY_PATH or
# engine construction fails with a bare "Unable to load library". Inference-only
# hosts never touch it, which is why the failure only appears at build time.
for so in "${SP}"/tensorrt_libs/libnvinfer_builder_resource*.so.*; do
    [[ -f "${so}" ]] && ln -sfn "${so}" "${TP}/tensorrt/lib/$(basename "${so}")"
done

# CUDA comes from the pip wheel too, on purpose: TensorRT 10.3's x86 build
# links libcudart.so.12, while this host's apt CUDA is 11.5. Loading two
# cudart majors into one process is undefined behaviour, so we compile and
# link against the same CUDA 12 the TensorRT binary already uses.
CUDA_PIP="${SP}/nvidia/cuda_runtime"
[[ -f "${CUDA_PIP}/include/cuda_runtime_api.h" ]] || die "pip CUDA runtime headers missing"
ln -sfn "${CUDA_PIP}/include" "${TP}/cuda/include"
ln -sfn "${CUDA_PIP}/lib/libcudart.so.12" "${TP}/cuda/lib/libcudart.so.12"
ln -sfn "${CUDA_PIP}/lib/libcudart.so.12" "${TP}/cuda/lib/libcudart.so"
say "cuda runtime $(basename "$(readlink -f "${CUDA_PIP}/lib/libcudart.so.12")")"

say "done -- now:  source ${DEPLOY_ROOT}/env.sh"
