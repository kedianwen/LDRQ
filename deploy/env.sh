# Source me:  source deploy/env.sh
#
# Puts the shell into the state the deployment workspace expects, on either
# host: an x86 dev box provisioned by setup.sh, or a Jetson where JetPack
# already supplies TensorRT and CUDA under /usr.

# shellcheck shell=bash

_r1_deploy_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Conda's python shadows the system python3.10 that ROS 2 humble was built
# against, and ament picks up whichever comes first on PATH. Step out of any
# active conda environment before sourcing ROS.
if [[ -n "${CONDA_PREFIX:-}" ]] && command -v conda >/dev/null 2>&1; then
    while [[ -n "${CONDA_PREFIX:-}" ]]; do conda deactivate; done
fi

source /opt/ros/humble/setup.bash

# Where the C++ build finds TensorRT and the CUDA runtime. On a Jetson both
# live in the JetPack system prefix; on this dev box they come from the venv
# via the link farms setup.sh stages.
if [[ -f "${_r1_deploy_root}/third_party/tensorrt/include/NvInfer.h" ]]; then
    export TENSORRT_ROOT="${_r1_deploy_root}/third_party/tensorrt"
    export CUDART_ROOT="${_r1_deploy_root}/third_party/cuda"
else
    export TENSORRT_ROOT="/usr"
    export CUDART_ROOT="/usr/local/cuda"
fi

export LD_LIBRARY_PATH="${TENSORRT_ROOT}/lib:${CUDART_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export R1_DEPLOY_ROOT="${_r1_deploy_root}"

# Overlay the workspace if it has been built.
if [[ -f "${_r1_deploy_root}/ros2_ws/install/setup.bash" ]]; then
    source "${_r1_deploy_root}/ros2_ws/install/setup.bash"
fi

echo "r1 deploy env: TENSORRT_ROOT=${TENSORRT_ROOT}  CUDART_ROOT=${CUDART_ROOT}  ROS=${ROS_DISTRO}"
unset _r1_deploy_root
