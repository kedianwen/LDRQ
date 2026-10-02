#!/usr/bin/env bash
# W08 on-robot preflight. Checks the things that have each cost this project a
# session, and that all fail QUIETLY -- the reason they cost a session is that
# none of them produces an error message.
#
#   export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1
#   cd <install dir>/deploy && source env.sh      (on the lab robot: ~/kdw_deploy)
#   bash $R1_DEPLOY_ROOT/tools/w08_preflight.sh                # before starting the stack
#   bash $R1_DEPLOY_ROOT/tools/w08_preflight.sh --lock-clocks  # and pin the power model
#   bash $R1_DEPLOY_ROOT/tools/w08_preflight.sh --live         # again, with it running
#
# Exit code is the number of FAILs. WARNs do not fail: some of them are
# judgement calls that only the operator standing next to the robot can settle.
#
# It reports and does not repair, with one opt-in exception: --lock-clocks pins
# the power model and clocks. That one is opt-in rather than automatic because it
# needs root and changes how the whole board behaves; everything else that would
# change what the robot is doing is printed as a command for you to run.

set -uo pipefail

# Count the processes that really ARE <name>, not merely mention it.
#
# Two steps because neither alone works. FIND with `pgrep -f`: comm is truncated
# to 15 characters, so `pgrep r1_hw_bridge_node` finds nothing while it runs.
# CONFIRM with comm: -f also matches any process carrying the name in its own
# argv -- a shell invoked with a command that mentions it, for one. Excluding $$
# and $PPID is not enough, since the offender can be a grandparent. A shell that
# only mentions the name has comm=bash; the node has comm equal to the name cut
# to 15 characters. This project has now been bitten by the -f false positive in
# three separate tools, which is why it is one function.
count_real() {
  local name=$1 want=${1:0:15} pid n=0
  for pid in $(pgrep -f "$name" 2>/dev/null); do
    [[ -r "/proc/$pid/comm" ]] || continue
    [[ "$(< "/proc/$pid/comm")" == "$want" ]] && n=$((n+1))
  done
  echo "$n"
}

FAIL=0
WARN=0
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$*"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; FAIL=$((FAIL+1)); }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$*"; WARN=$((WARN+1)); }
info() { printf '        %s\n' "$*"; }
head1() { printf '\n\033[1m%s\033[0m\n' "$*"; }

LIVE=0
LOCK=0
for a in "$@"; do
  case "$a" in
    --live)        LIVE=1 ;;
    --lock-clocks) LOCK=1 ;;
    -h|--help)     sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $a  (--live, --lock-clocks)"; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
head1 "0. second writer on rt/lowcmd"
# The question is NOT "is the factory process running". Measured on the robot
# 2026-09-25: developer mode does NOT stop
# /unitree/module/master_service/master_service -- it keeps a live pid after the
# handheld switch. So a name match answers yes forever, and a check that always
# fires is worse than no check: it teaches you to skip a safety gate.
#
# What matters is whether anything is PUBLISHING on the topic we publish on.
# probe_lowcmd measures that directly. Prefer it; fall back to the name match
# plus an activity sample, and say plainly that the fallback is a guess.
LOWCMD_PROBE=""
for c in "$R1_DEPLOY_ROOT/tools/probe_cpp/build/probe_lowcmd" \
         "$R1_DEPLOY_ROOT/tools/probe_cpp/probe_lowcmd"; do
  [[ -x "$c" ]] && { LOWCMD_PROBE="$c"; break; }
done

# CPU time a pid burns over a 1 s window, in percent of one core. Read from /proc
# rather than taken from `ps %cpu`, which reports the average since the process
# STARTED -- useless for a service that has been up for hours.
cpu_pct_1s() {
  local pid=$1 a b
  a=$(awk '{print $14+$15}' "/proc/$pid/stat" 2>/dev/null) || return 1
  sleep 1
  b=$(awk '{print $14+$15}' "/proc/$pid/stat" 2>/dev/null) || return 1
  # $14+$15 are utime+stime in clock ticks; USER_HZ is 100 here, so the delta
  # over one second already is a percentage of one core.
  echo $(( b - a ))
}

FACTORY=$(pgrep -a -f 'master_service|sport_mode|ai_sport|motion_switcher' 2>/dev/null \
          | grep -v preflight || true)

if [[ -n "$LOWCMD_PROBE" ]]; then
  info "measuring rt/lowcmd directly (3 s, read-only)..."
  if PROBE_OUT=$("$LOWCMD_PROBE" --seconds 3 2>&1); then
    pass "nothing is writing rt/lowcmd -- no second writer"
    [[ -n "$FACTORY" ]] && {
      info "(master_service IS running -- expected, it survives developer mode."
      info " What matters is that it is not publishing, and it is not.)"
    }
  else
    RC=$?
    if (( RC == 2 )); then
      warn "probe_lowcmd refused to measure:"
      printf '%s\n' "$PROBE_OUT" | tail -3 | sed 's/^/          /'
    else
      fail "SOMETHING ELSE IS WRITING rt/lowcmd -- do not start the stack"
      printf '%s\n' "$PROBE_OUT" | grep -E 'messages|rate:' | sed 's/^/          /'
      info "If the handheld says developer mode, the mode did not take. Switch"
      info "again and re-run. Two writers at 500 Hz overwrite each other, and the"
      info "symptoms all look like a sim2real gap."
    fi
  fi
else
  warn "probe_lowcmd not built -- falling back to a process-name guess"
  info "build it once, and this check becomes a measurement instead:"
  info "  bash \$R1_DEPLOY_ROOT/tools/probe_cpp/build.sh"
  if [[ -n "$FACTORY" ]]; then
    info ""
    info "a factory motion process is running:"
    printf '%s\n' "$FACTORY" | sed 's/^/          /'
    info "That alone is NOT a failure. Developer mode leaves master_service"
    info "running; only its WRITING matters. Sampling what it is doing (1 s)..."
    BUSY=0
    while read -r pid _; do
      [[ -n "$pid" ]] || continue
      PCT=$(cpu_pct_1s "$pid") || continue
      info "  pid $pid: ${PCT}% of one core"
      (( PCT >= 3 )) && BUSY=1
    done <<<"$FACTORY"
    if (( BUSY == 1 )); then
      fail "it is burning CPU like an active 500 Hz writer -- do not start the stack"
      info "switch to developer mode on the handheld and re-run."
    else
      warn "near-idle, which is what a released service looks like. PROXY ONLY."
      info "Confirm developer mode on the handheld, build probe_lowcmd, and watch"
      info "the first seconds of the stack for sway/grinding/tremor."
    fi
  else
    warn "no factory motion process matched by name -- weak evidence either way"
    info "confirm developer mode on the handheld."
  fi
fi
info ""
info "L2+B (damping) is a resting state only -- it also makes the factory service"
info "the active writer, so do not use it while running."

# ---------------------------------------------------------------------------
head1 "1. environment"
if [[ -z "${R1_DEPLOY_ROOT:-}" ]]; then
  fail "R1_DEPLOY_ROOT is empty -- env.sh was not sourced in THIS shell"
  info "source <install dir>/deploy/env.sh   (e.g. ~/kdw_deploy/deploy/env.sh)"
  info "Every relative path downstream then silently resolves from /, and the"
  info "error you get is 'No such file or directory', which reads as a missing"
  info "build rather than a missing variable."
  echo; echo "stopping here: nothing below is meaningful without it."
  exit $((FAIL))
fi
[[ -d "$R1_DEPLOY_ROOT" ]] && pass "R1_DEPLOY_ROOT=$R1_DEPLOY_ROOT" \
                           || fail "R1_DEPLOY_ROOT=$R1_DEPLOY_ROOT does not exist"

# env.sh sets these with ${VAR:-default}, which only fills an UNSET variable. A
# shell that already carries ROS_LOCALHOST_ONLY=0 keeps the 0, sourcing env.sh
# does not override it, and the two terminals then cannot see each other's
# nodes. 2026-09-24 cost 16% of the samples in two topics this way.
if [[ "${ROS_LOCALHOST_ONLY:-unset}" == "1" ]]; then
  pass "ROS_LOCALHOST_ONLY=1"
else
  fail "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-unset} (env.sh defaults it to 1 with"
  info "\${VAR:-1}, which does NOT override an inherited value)"
  info "fix, in EVERY terminal, BEFORE sourcing env.sh:"
  info "  export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1"
  info "  cd $(dirname "$(dirname "$(readlink -f "$0")")") && source env.sh"
  info "Symptom if ignored: nodes do not see each other. ros2 topic list comes"
  info "back short and nothing reports a reason."
fi
# Exactly the same ${VAR:-default} trap as above, and it is symmetric: an
# inherited ROS_DOMAIN_ID=0 survives sourcing env.sh just as an inherited
# ROS_LOCALHOST_ONLY=0 does. On 2026-09-24 the domain happened to be unset so it
# picked up the 99 and only localhost_only was wrong -- that was luck, not
# design. Set both explicitly and neither depends on luck.
if [[ "${ROS_DOMAIN_ID:-unset}" == "99" ]]; then
  pass "ROS_DOMAIN_ID=99"
else
  fail "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset}; env.sh defaults it to 99 with"
  info "\${VAR:-99}, which does NOT override an inherited value either"
  info "fix, in EVERY terminal, BEFORE sourcing env.sh:"
  info "  export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1"
  info "  cd $(dirname "$(dirname "$(readlink -f "$0")")") && source env.sh"
  info "Two terminals on different domains cannot see each other, and the"
  info "symptom is a short topic list, not an error."
fi
info "ROS_DISTRO=${ROS_DISTRO:-unset}  RMW=${RMW_IMPLEMENTATION:-unset}"

# ---------------------------------------------------------------------------
head1 "2. binaries: present, and executable"
BIN="$R1_DEPLOY_ROOT/ros2_ws/install/r1_policy_runner/lib/r1_policy_runner"
MISSING=0
for b in r1_build_engine r1_parity_check r1_policy_node; do
  if [[ ! -e "$BIN/$b" ]]; then
    fail "$b not built"
    MISSING=1
  elif [[ ! -x "$BIN/$b" ]]; then
    # An archive round-trip can drop the execute bit. Bash then says "No such
    # file or directory", and sudo says "command not found" -- both of which
    # read as "not built" rather than "not executable".
    fail "$b exists but is NOT executable"
    info "chmod +x $BIN/*"
    info "Without the bit, bash reports 'No such file or directory' and sudo"
    info "reports 'command not found'. Neither names the actual problem, and"
    info "sudo is the wrong fix."
  else
    pass "$b executable"
  fi
done
[[ $MISSING -eq 1 ]] && info "colcon build --packages-select r1_policy_runner"

# ---------------------------------------------------------------------------
head1 "3. stale install/ -- sources newer than what was built"
# After an overlay package is extracted, install/ still holds the previous
# build. The node runs the old code; the new status line simply does not appear,
# which reads as "no problem found" rather than "no measurement taken".
for pkg in r1_hw_bridge r1_policy_runner; do
  SRC="$R1_DEPLOY_ROOT/ros2_ws/src/$pkg"
  INST="$R1_DEPLOY_ROOT/ros2_ws/install/$pkg"
  [[ -d "$SRC" ]] || { warn "$pkg: no source tree"; continue; }
  if [[ ! -d "$INST" ]]; then warn "$pkg: never built"; continue; fi
  NEWEST_SRC=$(find "$SRC" -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.py' \
                -o -name '*.yaml' -o -name 'CMakeLists.txt' \) -printf '%T@ %p\n' 2>/dev/null \
                | sort -rn | head -1)
  NEWEST_INST=$(find "$INST" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1)
  TS_SRC=${NEWEST_SRC%% *}; TS_INST=${NEWEST_INST%% *}
  if [[ -z "$TS_SRC" || -z "$TS_INST" ]]; then warn "$pkg: cannot compare timestamps"; continue; fi
  if (( ${TS_SRC%.*} > ${TS_INST%.*} )); then
    fail "$pkg: source is NEWER than install/ -- rebuild before running"
    info "newest source:  ${NEWEST_SRC#* }"
    info "cd \$R1_DEPLOY_ROOT/ros2_ws && colcon build --packages-select $pkg && source install/setup.bash"
  else
    pass "$pkg: install/ is newer than its sources"
  fi
done

# ---------------------------------------------------------------------------
head1 "4. engines"
for f in policy_fp32.plan policy_fp16.plan policy_int8.plan; do
  P="$R1_DEPLOY_ROOT/artifacts/$f"
  if [[ -f "$P" ]]; then
    pass "$f  $(stat -c%s "$P") B"
  elif [[ "$f" == policy_int8.plan ]]; then
    info "policy_int8.plan absent -- expected until step 3 of the W08 doc builds it"
  else
    fail "$f missing"
  fi
done
FX="$R1_DEPLOY_ROOT/artifacts/parity_fixture.bin"
if [[ -f "$FX" ]]; then
  # The fixture is the anchor every parity number is quoted against. Replacing it
  # voids all of them at once, and nothing announces that.
  pass "parity_fixture.bin  $(stat -c%s "$FX") B  md5=$(md5sum "$FX" | cut -c1-12)..."
  info "if that md5 prefix ever changes, EVERY archived parity number is void"
else
  fail "parity_fixture.bin missing -- no parity claim can be made without it"
fi
if [[ -x "$BIN/r1_parity_check" ]]; then
  info "fingerprints (parity certifies a FILE, not the ONNX):"
  for f in policy_fp32.plan policy_fp16.plan policy_int8.plan; do
    P="$R1_DEPLOY_ROOT/artifacts/$f"
    [[ -f "$P" ]] || continue
    info "  $f  $(stat -c%s "$P")B  (run r1_parity_check for the fnv1a)"
  done
fi

# ---------------------------------------------------------------------------
head1 "5. power model and clocks"
# Not just a benchmark concern. An unpinned board also runs the 50 Hz control
# loop with the GPU dropping to a low-power state between inferences, so this
# belongs BEFORE the stack comes up, not only before the benchmark. W05 measured
# the latency tail tightening 11x from pinning alone.
# `nvpmodel -q` without root also tries to read /sys/kernel/nvpmodel_emc_cap/,
# which is root-only, and prints NVPM ERROR lines about it (errno 13 = EACCES).
# The MODE it reports is still correct. Those lines are stripped from the summary
# so they cannot hide a different error -- and counted, so they are not simply
# swallowed either. The EMC clock they failed to read is verified for real below.
nvpm_mode() {
  nvpmodel -q 2>&1 | grep -v 'NVPM ERROR' | tr '\n' ' ' | sed 's/  */ /g; s/ *$//'
}
nvpm_noise() {
  nvpmodel -q 2>&1 | grep -c 'NVPM ERROR' || true
}

CUR_MODE=""
MAXN_ID=""
if command -v nvpmodel >/dev/null 2>&1; then
  CUR_MODE=$(nvpm_mode)
  info "current: ${CUR_MODE:-unreadable}"
  NOISE=$(nvpm_noise)
  if [[ "${NOISE:-0}" -gt 0 ]]; then
    info "(nvpmodel also printed $NOISE 'NVPM ERROR' line(s) about"
    info " /sys/kernel/nvpmodel_emc_cap -- that is EACCES from querying without"
    info " root. It does not affect the mode. EMC is checked directly below.)"
  fi
  # Discover the mode IDs from the board's own config rather than assuming a
  # number. Mode numbering is per-module: 0 is MAXN on AGX Orin, but on an Orin
  # Nano 8GB mode 0 is 15W. Hard-coding `-m 0` can therefore LOWER the power cap
  # on some modules, which is the opposite of the intent and would show up only
  # as slower inference.
  if [[ -r /etc/nvpmodel.conf ]]; then
    MODES=$(grep -oP '<\s*POWER_MODEL\s+ID=\s*\K[0-9]+\s+NAME=\s*\S+' /etc/nvpmodel.conf \
            | sed 's/NAME=//' | sort -u)
    if [[ -n "$MODES" ]]; then
      info "modes on this module:"
      printf '%s\n' "$MODES" | sed 's/^/          /'
      MAXN_ID=$(printf '%s\n' "$MODES" | awk 'toupper($2) ~ /MAXN/ {print $1; exit}')
    fi
  fi
  if [[ -n "$MAXN_ID" ]]; then
    info "MAXN is mode $MAXN_ID on this module"
  else
    warn "no mode named MAXN found in /etc/nvpmodel.conf -- pick the highest-power"
    info "mode from the list above by hand. NOT guessing a number here on purpose."
  fi
else
  warn "nvpmodel not on PATH"
fi
GOV=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
GOV_BEFORE=""
[[ -r $GOV ]] && { GOV_BEFORE=$(cat $GOV); info "cpu0 governor=$GOV_BEFORE"; }

if [[ $LOCK -eq 1 ]]; then
  if [[ -z "$MAXN_ID" ]]; then
    fail "--lock-clocks asked for, but the MAXN mode id could not be determined"
    info "run it by hand with the id from the list above, then re-run this script"
  else
    info ""
    info "--lock-clocks: applying. sudo will ask for a password."
    # Order matters: nvpmodel resets the clock policy, so jetson_clocks has to
    # come second or it is undone immediately.
    if sudo nvpmodel -m "$MAXN_ID" </dev/null; then
      info "nvpmodel -m $MAXN_ID applied"
    else
      fail "nvpmodel -m $MAXN_ID failed"
    fi
    if sudo jetson_clocks </dev/null 2>/dev/null || sudo /usr/bin/jetson_clocks </dev/null 2>/dev/null; then
      info "jetson_clocks applied"
    else
      warn "jetson_clocks not found or failed; clocks are NOT pinned"
    fi
    # Verify rather than assume: nvpmodel can decline a switch (and on some
    # modules asks to reboot), and a declined switch is not loud.
    NEW_MODE=$(nvpm_mode)
    if grep -qi 'maxn' <<<"$NEW_MODE"; then
      pass "power model now: $NEW_MODE"
    else
      fail "power model is still '$NEW_MODE' -- the switch did not take"
      info "some modules ask to reboot before a mode change applies"
    fi

    # "jetson_clocks applied" is the tool's exit status, not a measurement. What
    # matters is whether the cores actually sit at their ceiling, so read the
    # frequencies back. W05 on this Orin after locking: 8 cores at 1984 MHz,
    # GPU 918 MHz, EMC 3199 MHz -- if the CPUs come back well under their own
    # reported maximum, the lock did not take however cleanly it exited.
    [[ -r $GOV ]] && info "cpu0 governor now=$(cat $GOV)  (was ${GOV_BEFORE:-?})"
    # Test scaling_min_freq, not scaling_cur_freq. What jetson_clocks DOES is pin
    # the floor to the ceiling, so min==max is the configuration it is supposed to
    # have left behind -- a deterministic read. scaling_cur_freq is an
    # instantaneous sample and an idle core dips below max even when correctly
    # pinned, so gating on it produces a FAIL that comes and goes. It is printed
    # alongside for context only.
    UNPINNED=0; NCPU=0
    for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq; do
      [[ -r $c/scaling_min_freq && -r $c/cpuinfo_max_freq ]] || continue
      MIN=$(cat "$c/scaling_min_freq"); MAX=$(cat "$c/cpuinfo_max_freq")
      CUR=$(cat "$c/scaling_cur_freq" 2>/dev/null || echo 0)
      NCPU=$((NCPU+1))
      (( MIN < MAX )) && UNPINNED=$((UNPINNED+1))
      info "  $(basename "$(dirname "$c")")  min $((MIN/1000)) / max $((MAX/1000)) MHz  (now $((CUR/1000)))"
    done
    if (( NCPU == 0 )); then
      warn "no readable cpufreq nodes; cannot verify the CPU clocks"
    elif (( UNPINNED == 0 )); then
      pass "all $NCPU online cores have their floor pinned to their ceiling"
    else
      fail "$UNPINNED of $NCPU cores still have min < max -- clocks are NOT pinned"
      info "jetson_clocks exited cleanly but left the floor down. Re-run it, and"
      info "check nothing else (a thermal cap, another governor writer) is fighting it."
    fi
    # GPU and EMC live in module-specific paths, so print whatever jetson_clocks
    # reports rather than guessing at sysfs nodes that differ per JetPack.
    if JC=$(sudo jetson_clocks --show </dev/null 2>/dev/null); then
      grep -Ei 'GPU|EMC|^SOC|Thermal' <<<"$JC" | head -8 | sed 's/^/        /'
    else
      info "jetson_clocks --show unavailable; GPU/EMC not verified here"
    fi

    info "jetson_clocks does NOT survive a reboot. Re-run this after every"
    info "power cycle, or the latency numbers silently change under you."
  fi
else
  warn "clocks not pinned. Do it before the stack comes up, not just before the"
  info "benchmark -- a low-power GPU state between 20 ms-apart inferences is a"
  info "control-loop concern too:"
  info "  bash \$R1_DEPLOY_ROOT/tools/w08_preflight.sh --lock-clocks"
  info "(or by hand: sudo nvpmodel -m ${MAXN_ID:-<MAXN id above>} && sudo jetson_clocks)"
fi

# ---------------------------------------------------------------------------
head1 "6. duplicate nodes -- the reason 09-04 measured 100 Hz on a 50 Hz loop"
# The EXECUTABLE names, not the package names: `pgrep -f r1_hw_bridge` would also
# match any shell whose command line mentions the package -- this script's, for
# one. count_real() handles that; see its comment.
for pat in r1_hw_bridge_node r1_policy_node; do
  N=$(count_real "$pat")
  if [[ $LIVE -eq 1 ]]; then
    if   (( N == 0 )); then fail "$pat: not running (--live expects the stack up)"
    elif (( N == 1 )); then pass "$pat: exactly 1 process"
    else fail "$pat: $N processes -- the policy infers more than once per observation"
         info "pkill -f $pat   (then start ONE stack)"
    fi
  else
    (( N == 0 )) && pass "$pat: not running yet" \
                 || warn "$pat: $N process(es) already running; kill them before starting a new stack"
  fi
done

# ---------------------------------------------------------------------------
head1 "7. ros2 CLI daemon"
# foxy caches a graph snapshot. A stale one shows an empty topic list whether or
# not anything is wrong, which is indistinguishable from a dead bridge.
if command -v ros2 >/dev/null 2>&1; then
  ros2 daemon stop >/dev/null 2>&1 && pass "daemon stopped (stale graph cache cleared)" \
                                   || info "no daemon was running"
else
  warn "ros2 CLI not on PATH in this shell"
fi

# ---------------------------------------------------------------------------
head1 "8. room to write"
OUT="${HOME}/orin_commissioning"
mkdir -p "$OUT" 2>/dev/null
if [[ -d "$OUT" && -w "$OUT" ]]; then
  AVAIL=$(df -Pm "$OUT" | awk 'NR==2{print $4}')
  # 60 s of 425-float observations at 50 Hz is ~5 MB as R1CB and several times
  # that as the JSON walk_metrics writes.
  if (( AVAIL > 500 )); then pass "$OUT writable, ${AVAIL} MB free"
  else fail "$OUT has only ${AVAIL} MB free; a 60 s recording needs ~50 MB"; fi
else
  fail "$OUT is not writable"
fi

# ---------------------------------------------------------------------------
if [[ $LIVE -eq 1 ]]; then
  head1 "9. live topics"
  if ! command -v ros2 >/dev/null 2>&1; then
    warn "ros2 CLI unavailable; cannot list topics"
  else
    T=$(ros2 topic list 2>/dev/null)
    # /parameter_events and /rosout belong to the CLI's OWN node, so a list
    # containing only those means "I can see nothing but myself". That is a
    # discovery or liveness problem, and it looks exactly like "the bridge is not
    # publishing" -- which is why it has cost this project time three times. The
    # two cases need different answers, so separate them before concluding
    # anything about any individual topic.
    VISIBLE=$(grep -vx -e /parameter_events -e /rosout <<<"$T" | grep -c . || true)
    if (( VISIBLE == 0 )); then
      fail "the graph is EMPTY apart from the CLI's own /parameter_events + /rosout"
      info "This is NOT evidence that the bridge is not publishing -- it is"
      info "evidence that nothing was discovered. Three things do that, and the"
      info "block below tells you which:"
      info ""
      info "  a) the stack is not running any more:"
      if (( $(count_real r1_hw_bridge_node) > 0 )); then
        info "     -> r1_hw_bridge_node IS running. Not this."
      else
        info "     -> r1_hw_bridge_node is NOT running. THIS IS IT. The launch"
        info "        holds the terminal, so if you stopped it to type a command,"
        info "        the stack went with it. Start it in one terminal and leave"
        info "        it; run tools from a second one."
      fi
      info ""
      info "  b) this shell and the node disagree on a ROS variable:"
      NPID=""
      for cand in $(pgrep -f r1_hw_bridge_node 2>/dev/null); do
        [[ -r "/proc/$cand/comm" ]] || continue
        [[ "$(< "/proc/$cand/comm")" == "r1_hw_bridge_no" ]] && { NPID="$cand"; break; }
      done
      if [[ -n "$NPID" && -r "/proc/$NPID/environ" ]]; then
        NENV=$(tr '\0' '\n' < "/proc/$NPID/environ")
        for v in ROS_DOMAIN_ID ROS_LOCALHOST_ONLY RMW_IMPLEMENTATION; do
          THEIRS=$(sed -n "s/^${v}=//p" <<<"$NENV" | head -1)
          MINE="${!v-}"
          if [[ "$THEIRS" == "$MINE" ]]; then
            info "     $v: both '${MINE:-<unset>}'  ok"
          else
            info "     $v: MISMATCH -- this shell '${MINE:-<unset>}', node '${THEIRS:-<unset>}'"
            info "        RMW especially: this robot carries the SDK's CycloneDDS as"
            info "        well as foxy's, and two different RMWs never discover each"
            info "        other however right the domain is."
          fi
        done
      else
        info "     (no running node to compare against -- see (a))"
      fi
      info ""
      info "  c) a ros2 daemon started under a different environment is answering"
      info "     for you instead of discovering:"
      if pgrep -af _ros2_daemon >/dev/null 2>&1; then
        pgrep -af _ros2_daemon | sed 's/^/          /'
        info "     -> a daemon is running. Kill its cached view: ros2 daemon stop"
      else
        info "     -> no daemon running. Not this."
      fi
    else
      for t in /r1_hw_bridge/obs /r1_hw_bridge/joint_pos /r1_hw_bridge/cmd_debug \
               /r1_hw_bridge/status /r1_policy_runner/action; do
        grep -qx "$t" <<<"$T" && pass "$t" || fail "$t absent"
      done
      # Only meaningful once the rest of the graph IS visible: then a missing
      # joint_tau really does mean the bridge binary predates the overlay, and
      # walk_metrics will fall back to the kKp*error ESTIMATE.
      if grep -qx /r1_hw_bridge/joint_tau <<<"$T"; then
        pass "/r1_hw_bridge/joint_tau (measured torque available)"
      else
        fail "/r1_hw_bridge/joint_tau absent while the rest of the graph is visible"
        info "=> the bridge was not rebuilt from the W08 sources. walk_metrics.py"
        info "   will fall back to the kKp*error estimate. Rebuild:"
        info "  colcon build --packages-select r1_hw_bridge && source install/setup.bash"
      fi
    fi
  fi
fi

# ---------------------------------------------------------------------------
printf '\n\033[1m%d FAIL, %d WARN\033[0m\n' "$FAIL" "$WARN"
if (( FAIL > 0 )); then
  echo "Do not start the stack until the FAILs are cleared. Every one of them"
  echo "produces a plausible wrong answer rather than an error."
else
  echo "Preflight clear. Developer mode is still yours to confirm on the handheld."
fi
exit $FAIL
