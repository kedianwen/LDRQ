#!/usr/bin/env bash
# Install a policy bundle on the ROBOT. One command, eight checked steps.
#
#   bash install_bundle.sh <bundle_dir> [--dry-run] [--force-engine]
#
# Order matters: everything that can be decided from files is decided BEFORE the
# 29 s colcon build and the 20 s engine build, so a bad bundle costs seconds
# rather than a minute and a half. And the last thing that happens is a numeric
# parity check against the fixture the bundle brought with it -- a new policy is
# proved to compute the right numbers on this Orin before it is ever allowed to
# move a joint.
#
# Why this regenerates a C++ header and rebuilds instead of reading parameters at
# run time: joint_map.hpp's arrays are `constexpr`, so the compiler checks their
# length against kNumJoints/kNumActions and a mismatch is a build error rather
# than a run-time surprise on a robot that is already standing. The rebuild is
# measured at 29 s on this Orin (both packages). That is cheaper than the risk of
# changing the one node that has been stable since W06.
#
# NOT regenerated, deliberately: the unitree_hg slot map is a property of the
# robot, established by measurement, and it stays in the header's measured tsv.
set -uo pipefail

BUNDLE="${1:-}"
shift || true
DRY=0; FORCE_ENGINE=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --force-engine) FORCE_ENGINE=1 ;;
    *) echo "unknown argument: $a"; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
step=0
say() { step=$((step+1)); printf '\n[%d/8] %s\n' "$step" "$1"; }
die() { printf '\n[FAIL] %s\n' "$1" >&2; exit 1; }

[[ -n "$BUNDLE" && -d "$BUNDLE" ]] || die "usage: bash install_bundle.sh <bundle_dir> [--dry-run] [--force-engine]"
BUNDLE="$(cd "$BUNDLE" && pwd)"
: "${R1_DEPLOY_ROOT:?R1_DEPLOY_ROOT is not set -- source deploy/env.sh first}"
[[ -d "$R1_DEPLOY_ROOT" ]] || die "R1_DEPLOY_ROOT=$R1_DEPLOY_ROOT does not exist"

WS="$R1_DEPLOY_ROOT/ros2_ws"
ART="$R1_DEPLOY_ROOT/artifacts"
IFACE="$R1_DEPLOY_ROOT/interface"
PROBE="$R1_DEPLOY_ROOT/tools/probe_cpp"
HDR="$WS/src/r1_hw_bridge/include/r1_hw_bridge/joint_map.hpp"
PLAN="$ART/policy_fp32.plan"

echo "bundle : $BUNDLE"
echo "deploy : $R1_DEPLOY_ROOT"
[[ $DRY -eq 1 ]] && echo "MODE   : dry run -- nothing is written, nothing is built"

# ---------------------------------------------------------------- 1. validate
say "validate the bundle (no writes)"
python3 "$HERE/verify_bundle.py" "$BUNDLE" || die "the bundle is invalid; nothing was touched"

# ---------------------------------------------------------------- 2. preview
say "what would change in the deploy tree"
python3 "$HERE/gen_configs.py" "$BUNDLE" --deploy "$R1_DEPLOY_ROOT" --check
RUN_ID="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['run_id'])" "$BUNDLE/provenance.json")"
NEW_ONNX_SHA="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['onnx_sha256'])" "$BUNDLE/provenance.json")"
OLD_ONNX_SHA="$(python3 -c "
import json,pathlib,sys
p=pathlib.Path(sys.argv[1])
print(json.loads(p.read_text())['onnx_sha256'] if p.is_file() else '')" "$ART/installed.json" 2>/dev/null)"
echo "run id               : $RUN_ID"
if [[ -n "$OLD_ONNX_SHA" ]]; then
  echo "onnx sha256          : ${NEW_ONNX_SHA:0:12}  (installed: ${OLD_ONNX_SHA:0:12})"
else
  echo "onnx sha256          : ${NEW_ONNX_SHA:0:12}  (no installed.json -- first bundle install)"
fi

if [[ $DRY -eq 1 ]]; then
  printf '\ndry run complete. Re-run without --dry-run to install.\n'; exit 0
fi

# ---------------------------------------------------------------- 3. stage
say "stage the bundle into the deploy tree"
mkdir -p "$IFACE" "$ART"
# command_envelope.json goes into interface/ too: mission_ctl reads the envelope
# from there first, so the capability statement and the bridge's clamp come from
# the same installed bundle.
for f in policy_interface.json actuator_gains.json command_envelope.json; do
  cp -v "$BUNDLE/$f" "$IFACE/$f" || die "cannot stage $f"
done
cp -v "$BUNDLE/policy.onnx" "$ART/policy.onnx" || die "cannot stage policy.onnx"
cp -v "$BUNDLE/parity_fixture.bin" "$ART/parity_fixture.bin" || die "cannot stage the fixture"

# ------------------------------------------------------- 4. regenerate sources
say "regenerate joints.tsv and joint_map.hpp from the staged spec"
python3 "$PROBE/gen_joints.py" || die "gen_joints.py failed"
python3 "$PROBE/gen_joint_map.py" --header="$HDR" "$PROBE/joint_map_measured.tsv" \
  | tail -6 || die "gen_joint_map.py failed (this is where a gains/joint order mismatch surfaces)"

say "regenerate the two ROS configs"
python3 "$HERE/gen_configs.py" "$BUNDLE" --deploy "$R1_DEPLOY_ROOT" \
  || die "gen_configs.py failed"

# ---------------------------------------------------------------- 5. build C++
say "colcon build (r1_hw_bridge, r1_policy_runner)"
( cd "$WS" && colcon build --packages-select r1_hw_bridge r1_policy_runner ) \
  || die "colcon build failed -- the generated header did not compile, which is
       exactly what constexpr arrays are for. Nothing was installed."
# shellcheck disable=SC1091
source "$WS/install/setup.bash" || die "cannot source the freshly built overlay"

# ------------------------------------------------------------- 6. build engine
say "build the TensorRT engine on THIS machine"
if [[ "$NEW_ONNX_SHA" == "$OLD_ONNX_SHA" && -f "$PLAN" && $FORCE_ENGINE -eq 0 ]]; then
  echo "the ONNX is byte-identical to the installed one and $PLAN exists."
  echo "reusing it. Pass --force-engine to rebuild anyway."
else
  ros2 run r1_policy_runner r1_build_engine \
    --onnx "$ART/policy.onnx" --plan "$PLAN" \
    || die "engine build failed"
fi

# -------------------------------------------------------------- 7. parity gate
say "parity check: this engine against the bundle's own fixture"
ros2 run r1_policy_runner r1_parity_check \
  --plan "$PLAN" --fixture "$ART/parity_fixture.bin" --tol 1e-3 \
  || die "PARITY FAILED. The engine on this machine does not reproduce the
       exported policy. Do NOT run it. Check that TF32 was cleared and that the
       fixture belongs to this ONNX."

# ------------------------------------------------------------------ 8. record
say "record what is installed"
FP="$(ros2 run r1_policy_runner r1_parity_check --plan "$PLAN" \
        --fixture "$ART/parity_fixture.bin" 2>/dev/null \
        | grep -o 'fnv1a=0x[0-9a-f]*' | head -1)"
python3 - "$ART/installed.json" "$BUNDLE" "$PLAN" "${FP:-unknown}" <<'PY'
import datetime, hashlib, json, pathlib, socket, sys
out, bundle, plan, fp = (pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]),
                         pathlib.Path(sys.argv[3]), sys.argv[4])
prov = json.loads((bundle / "provenance.json").read_text())
man = hashlib.sha256((bundle / "MANIFEST.sha256").read_bytes()).hexdigest()
out.write_text(json.dumps({
    "installed": datetime.datetime.now().isoformat(timespec="seconds"),
    "installed_on": socket.gethostname(),
    "run_id": prov["run_id"],
    "onnx_sha256": prov["onnx_sha256"],
    "bundle_manifest_sha256": man,
    "bundle_path": str(bundle),
    "plan": str(plan),
    "plan_bytes": plan.stat().st_size if plan.is_file() else None,
    "plan_fingerprint": fp,
}, indent=2) + "\n")
print("wrote", out)
PY

cat <<EOF

=========================================================================
installed: $RUN_ID
engine   : $PLAN  $FP
=========================================================================
Launch it (output OFF -- turning it on is a separate, deliberate act):

  ros2 launch r1_hw_bridge r1_stack.launch.py \\
      engine:=$PLAN iface:=eth10 enable_output:=false

Before enable_output:=true, remember the two things a script cannot check:
  - the handheld must be in DEVELOPER MODE, or the factory motion service
    keeps writing rt/lowcmd against you at 500 Hz;
  - clocks locked (tools/w08_preflight.sh --lock-clocks), or every latency
    number is a lottery.
EOF
