#!/usr/bin/env bash
# W08 on-robot preflight. Checks the things that have each cost this project a
# session, and that all fail QUIETLY -- the reason they cost a session is that
# none of them produces an error message.
#
#   source $R1_DEPLOY_ROOT/env.sh
#   bash $R1_DEPLOY_ROOT/tools/w08_preflight.sh          # before starting the stack
#   bash $R1_DEPLOY_ROOT/tools/w08_preflight.sh --live   # again, with it running
#
# Exit code is the number of FAILs. WARNs do not fail: some of them are
# judgement calls that only the operator standing next to the robot can settle.
#
# It reports and does not repair. Everything that would need root (nvpmodel,
# jetson_clocks) or that changes what the robot is doing is printed as a command
# for you to run, deliberately.

set -uo pipefail

FAIL=0
WARN=0
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$*"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; FAIL=$((FAIL+1)); }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$*"; WARN=$((WARN+1)); }
info() { printf '        %s\n' "$*"; }
head1() { printf '\n\033[1m%s\033[0m\n' "$*"; }

LIVE=0
[[ "${1:-}" == "--live" ]] && LIVE=1

# ---------------------------------------------------------------------------
head1 "0. developer mode -- the one check a script cannot make for you"
# The factory motion service is a 500 Hz writer on rt/lowcmd. With it running,
# two writers fight and the symptoms (torso sway, joint grinding, high-frequency
# tremor, stance not held) all look exactly like a sim2real gap. W06 lost a
# session to this.
FACTORY=$(pgrep -a -f 'master_service|sport_mode|ai_sport|motion_switcher' 2>/dev/null | grep -v preflight || true)
if [[ -n "$FACTORY" ]]; then
  fail "a factory motion process appears to be running:"
  printf '%s\n' "$FACTORY" | sed 's/^/          /'
  info "switch to developer mode on the handheld BEFORE starting the stack."
  info "L2+B (damping) is a resting state only -- it also makes the factory"
  info "service the active writer, so do not use it while running."
else
  pass "no factory motion process matched by name"
  warn "that is weak evidence. Name matching is not mode checking -- confirm"
  info "developer mode on the handheld anyway. (The real fix is a"
  info "MotionSwitcherClient::CheckMode() gate in the bridge; not built yet.)"
fi

# ---------------------------------------------------------------------------
head1 "1. environment"
if [[ -z "${R1_DEPLOY_ROOT:-}" ]]; then
  fail "R1_DEPLOY_ROOT is empty -- env.sh was not sourced in THIS shell"
  info "source ~/kdw_deploy/deploy/env.sh"
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
  info "fix, in EVERY terminal:  export ROS_LOCALHOST_ONLY=1 && source \$R1_DEPLOY_ROOT/env.sh"
  info "Symptom if ignored: nodes do not see each other. ros2 topic list comes"
  info "back short and nothing reports a reason."
fi
if [[ "${ROS_DOMAIN_ID:-unset}" == "99" ]]; then
  pass "ROS_DOMAIN_ID=99"
else
  warn "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset}, env.sh's default is 99 -- fine if"
  info "deliberate, but it must match in every terminal"
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
head1 "5. clocks -- unpinned clocks make every latency number a lottery"
if command -v nvpmodel >/dev/null 2>&1; then
  info "$(nvpmodel -q 2>/dev/null | tr '\n' ' ')"
else
  warn "nvpmodel not on PATH"
fi
GOV=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
[[ -r $GOV ]] && info "cpu0 governor=$(cat $GOV)"
warn "pin them before any benchmark (W05: locking cut the tail 11x):"
info "  sudo nvpmodel -m 0 && sudo jetson_clocks"
info "These need a password, so they are printed rather than run."

# ---------------------------------------------------------------------------
head1 "6. duplicate nodes -- the reason 09-04 measured 100 Hz on a 50 Hz loop"
# The EXECUTABLE names, not the package names: `pgrep -f r1_hw_bridge` also
# matches any shell whose own command line happens to mention the package -- this
# script's, for one. And -f is required either way, because comm is truncated to
# 15 characters, so a bare `pgrep r1_hw_bridge_node` finds nothing while it runs.
for pat in r1_hw_bridge_node r1_policy_node; do
  # Excluding this script and its parent shell, which can carry the pattern in
  # their own argv and would otherwise count as a running node.
  N=$(pgrep -f "$pat" 2>/dev/null | grep -vx -e "$$" -e "$PPID" | wc -l)
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
  if command -v ros2 >/dev/null 2>&1; then
    T=$(ros2 topic list 2>/dev/null)
    for t in /r1_hw_bridge/obs /r1_hw_bridge/joint_pos /r1_hw_bridge/cmd_debug \
             /r1_hw_bridge/status /r1_policy_runner/action; do
      grep -qx "$t" <<<"$T" && pass "$t" || fail "$t absent"
    done
    # New in the W08 package. Its absence means the bridge binary predates the
    # overlay, so walk_metrics falls back to the kKp*error ESTIMATE and reports
    # an estimate as though it were a measurement unless it is told.
    if grep -qx /r1_hw_bridge/joint_tau <<<"$T"; then
      pass "/r1_hw_bridge/joint_tau (measured torque available)"
    else
      fail "/r1_hw_bridge/joint_tau absent -- bridge was not rebuilt from the W08 sources"
      info "walk_metrics.py will fall back to the kKp*error estimate. Rebuild:"
      info "  colcon build --packages-select r1_hw_bridge && source install/setup.bash"
    fi
  else
    warn "ros2 CLI unavailable; cannot list topics"
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
