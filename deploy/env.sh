# Source me:  source deploy/env.sh
#
# Puts the shell into the state the deployment workspace expects, on either
# host: an x86 dev box provisioned by setup.sh, or a Jetson where JetPack
# already supplies TensorRT and CUDA under /usr.

# shellcheck shell=bash

_r1_deploy_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Conda's python shadows the system python that ROS 2 was built against, and
# ament picks up whichever comes first on PATH. Step out of any active conda
# environment before sourcing ROS.
# Bounded, not `while`: in a non-interactive shell `conda` may resolve to the
# launcher script rather than the shell function, in which case `deactivate`
# cannot unset CONDA_PREFIX in this shell and an unbounded loop spins forever.
if [[ -n "${CONDA_PREFIX:-}" ]] && command -v conda >/dev/null 2>&1; then
    for _ in 1 2 3 4 5; do
        [[ -z "${CONDA_PREFIX:-}" ]] && break
        conda deactivate >/dev/null 2>&1 || break
    done
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        echo "r1 deploy env: WARNING could not leave the conda env (CONDA_PREFIX=${CONDA_PREFIX})." \
             "Run 'conda deactivate' by hand in an interactive shell before building." >&2
    fi
fi

# ROS 2 distro. Do NOT hardcode: this dev box has humble, but the R1's Orin
# ships JetPack 5.1.1 / Ubuntu 20.04, where the ROS 2 distro is foxy (and a
# ROS 1 noetic sits alongside it). Honour an explicit R1_ROS_DISTRO, else take
# the newest ROS 2 distro present.
if [[ -n "${R1_ROS_DISTRO:-}" ]]; then
    _r1_ros_candidates=("${R1_ROS_DISTRO}")
else
    # Newest first. noetic is ROS 1 and is deliberately absent from this list:
    # sourcing it into the same shell as ROS 2 breaks both.
    _r1_ros_candidates=(jazzy iron humble galactic foxy)
fi

_r1_ros_found=""
for _d in "${_r1_ros_candidates[@]}"; do
    if [[ -f "/opt/ros/${_d}/setup.bash" ]]; then _r1_ros_found="${_d}"; break; fi
done

if [[ -z "${_r1_ros_found}" ]]; then
    echo "r1 deploy env: ERROR no ROS 2 distro found under /opt/ros (looked for:" \
         "${_r1_ros_candidates[*]}). Set R1_ROS_DISTRO=<name> and re-source." >&2
else
    # A ROS 1 setup.bash already sourced into this shell poisons the ROS 2
    # build with ROS_PACKAGE_PATH / CMAKE_PREFIX_PATH entries that colcon then
    # tries to interpret as ament packages.
    if [[ -n "${ROS_PACKAGE_PATH:-}" ]]; then
        echo "r1 deploy env: WARNING ROS_PACKAGE_PATH is set (a ROS 1 setup.bash was" \
             "sourced here). Start a clean shell before building." >&2
    fi
    source "/opt/ros/${_r1_ros_found}/setup.bash"
fi
unset _r1_ros_candidates _r1_ros_found _d

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

# lib64 as well as lib: JetPack puts the CUDA runtime under
# /usr/local/cuda/lib64 (a symlink into targets/aarch64-linux/lib), while the
# pip wheels staged by setup.sh use lib.
# /usr/local/lib: unitree_sdk2 and the CycloneDDS it ships install there. The
# static libunitree_sdk2.a needs no runtime path, but the DDS libraries beside
# it may be shared depending on how the SDK was built.
export LD_LIBRARY_PATH="${TENSORRT_ROOT}/lib:${CUDART_ROOT}/lib:${CUDART_ROOT}/lib64:/usr/local/lib:${LD_LIBRARY_PATH:-}"

# W06: the bridge links unitree_sdk2, which statically contains CycloneDDS
# 0.10.2. Loading ROS's rmw_cyclonedds_cpp into the same process pulls in the
# system CycloneDDS 0.7.0 as well -- two versions of the same library in one
# address space. Pin the ROS side to Fast-RTPS (also foxy's default) so it never
# happens by accident.
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

# Isolate the ROS side from the robot network by DEFAULT, not by remembering to.
# Domain 0 on this machine carries the factory stack's DDS traffic; a ROS
# participant that joins it dies in discovery -- `ros2 topic list` and any rclpy
# node throw std::bad_alloc and get OOM-killed (measured 2026-08-26 and again
# 2026-09-02). The symptom is "topic does not appear to be published yet",
# which reads as a missing publisher rather than a poisoned domain.
#
# This does NOT touch the robot link: the bridge reaches rt/lowstate through
# unitree_sdk2's own CycloneDDS participant, bound to its own interface and
# domain. Observed: obs stayed at 50.0 Hz with the ROS side on domain 99.
#
# Override both if you deliberately need to talk to something off-box.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-99}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"

# `:-` only fills in an UNSET variable. A shell that already carries
# ROS_LOCALHOST_ONLY=0 keeps it, and the default above is silently bypassed --
# which is how two terminals ended up on 1 and 0 and stopped seeing each other
# (2026-09-02). Discovery just fails; nothing reports a reason. So say it.
if [[ "${ROS_LOCALHOST_ONLY}" != "1" ]]; then
    echo "WARNING: ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY} (inherited, not the default 1)." >&2
    echo "         On this robot every terminal must agree, or ROS nodes will not" >&2
    echo "         discover each other and \`ros2 topic list\` comes back empty." >&2
    echo "         Fix with: export ROS_LOCALHOST_ONLY=1 && source env.sh" >&2
fi
export R1_DEPLOY_ROOT="${_r1_deploy_root}"

# Cross-terminal check.
#
# Three times now (2026-09-02, 09-24, 09-25) a second terminal could not see the
# running stack because its ROS variables differed from the terminal the stack
# was launched in. The failure is silent: `ros2 topic list` comes back with only
# the CLI's own /parameter_events and /rosout, which looks identical to "the
# bridge is not publishing", and nothing anywhere names the cause.
#
# The check above can only inspect THIS shell, and the mismatch is by definition
# between two shells. So compare against what the RUNNING node actually has, read
# out of /proc/<pid>/environ. That works precisely when DDS discovery does not --
# the filesystem does not care about domains -- which is what makes the check
# possible at all.
_r1_check_running_stack() {
    local pid v mine theirs differs=0 cand
    # Two steps, because neither alone is correct.
    #
    # FIND with -f: comm is truncated to 15 characters, so `pgrep
    # r1_hw_bridge_node` finds nothing while the node is running.
    #
    # CONFIRM with comm: -f matches any process merely CARRYING the name in its
    # argv -- a shell that was invoked with a command mentioning it, for
    # instance, which is a real false positive and has bitten this project three
    # times now in three different tools. Excluding $$ and $PPID is not enough:
    # the offender can be a grandparent. But a shell that only mentions the name
    # has comm=bash, while the node has comm equal to the name truncated to 15
    # characters. That is the discriminator.
    local want_comm="r1_hw_bridge_no"          # first 15 chars of r1_hw_bridge_node
    pid=""
    for cand in $(pgrep -f r1_hw_bridge_node 2>/dev/null); do
        [[ -r "/proc/${cand}/comm" ]] || continue
        if [[ "$(< "/proc/${cand}/comm")" == "${want_comm}" ]]; then
            pid="${cand}"
            break
        fi
    done
    [[ -n "${pid}" && -r "/proc/${pid}/environ" ]] || return 0

    # One read of the environ, reused: it is a NUL-separated blob, so translate
    # once rather than per variable.
    local env_lines
    env_lines=$(tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null) || return 0

    local node_domain node_localhost
    node_domain=$(sed -n 's/^ROS_DOMAIN_ID=//p' <<< "${env_lines}" | head -1)
    node_localhost=$(sed -n 's/^ROS_LOCALHOST_ONLY=//p' <<< "${env_lines}" | head -1)

    for v in ROS_DOMAIN_ID ROS_LOCALHOST_ONLY RMW_IMPLEMENTATION; do
        theirs=$(sed -n "s/^${v}=//p" <<< "${env_lines}" | head -1)
        mine="${!v-}"
        [[ -z "${theirs}" && -z "${mine}" ]] && continue
        if [[ "${theirs}" != "${mine}" ]]; then
            if [[ ${differs} -eq 0 ]]; then
                echo "WARNING: a bridge is already running (pid ${pid}) with DIFFERENT ROS" >&2
                echo "         settings. These two terminals will NOT see each other's nodes," >&2
                echo "         and \`ros2 topic list\` will show only its own two topics." >&2
                differs=1
            fi
            printf '           %-20s this shell: %-12s running node: %s\n' \
                   "${v}" "${mine:-<unset>}" "${theirs:-<unset>}" >&2
        fi
    done
    if [[ ${differs} -eq 1 ]]; then
        echo "         Match the running node, then re-source and clear the CLI cache:" >&2
        echo "           export ROS_DOMAIN_ID=${node_domain:-99} ROS_LOCALHOST_ONLY=${node_localhost:-1}" >&2
        echo "           cd ${_r1_deploy_root} && source env.sh && ros2 daemon stop" >&2
    fi
}
_r1_check_running_stack

# Overlay the workspace if it has been built.
if [[ -f "${_r1_deploy_root}/ros2_ws/install/setup.bash" ]]; then
    source "${_r1_deploy_root}/ros2_ws/install/setup.bash"
fi

echo "r1 deploy env: TENSORRT_ROOT=${TENSORRT_ROOT}  CUDART_ROOT=${CUDART_ROOT}  ROS=${ROS_DISTRO}"
# Print the values that actually took effect, not the ones we tried to set:
# every terminal has to match, and a mismatch is invisible until discovery
# quietly fails.
echo "               ROS_DOMAIN_ID=${ROS_DOMAIN_ID}  ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY}  RMW=${RMW_IMPLEMENTATION}"
unset _r1_deploy_root
