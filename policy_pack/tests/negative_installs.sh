#!/usr/bin/env bash
# Stage D acceptance on the ROBOT: three bad bundles through the REAL installer.
#
#   bash negative_installs.sh <good_bundle_dir>
#
# run_tests.sh proves verify_bundle.py refuses bad bundles. This proves the
# thing the operator actually runs, install_bundle.sh (no --dry-run), stops at
# step 1 -- before it stages a file, and long before the colcon build -- names
# the reason, and leaves the deploy tree byte-for-byte as it was. The three
# cases are the stage D plan's:
#   1. history_length 5 -> 6 in policy_interface.json
#   2. actuator_gains.json missing one joint
#   3. an observation term the bridge cannot compute (base_lin_vel)
# Exit code = number of cases that did not behave.
set -uo pipefail
GOOD="${1:-}"
[[ -n "$GOOD" && -f "$GOOD/MANIFEST.sha256" ]] || { echo "usage: bash negative_installs.sh <good_bundle_dir>"; exit 2; }
GOOD="$(cd "$GOOD" && pwd)"
: "${R1_DEPLOY_ROOT:?R1_DEPLOY_ROOT is not set -- source deploy/env.sh first}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL="$(dirname "$HERE")/install_bundle.sh"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

tree_hash() {  # every file the installer could touch, build output excluded
  ( cd "$R1_DEPLOY_ROOT" && find . -path ./ros2_ws/build -prune -o -path ./ros2_ws/log -prune \
      -o -path ./ros2_ws/install -prune -o -type f -print0 | sort -z | xargs -0 sha256sum ) | sha256sum | cut -c1-16
}

bad=0
case_install() {  # case_install <name> <expected reason> <python mutation>
  local name="$1" want="$2" code="$3" dir="$WORK/$RANDOM"
  cp -r "$GOOD" "$dir"
  ( cd "$dir" && python3 -c "$code" ) || { echo "SETUP FAILED: $name"; bad=$((bad+1)); return; }
  ( cd "$dir" && find . -type f ! -name MANIFEST.sha256 -printf '%P\n' | sort | xargs sha256sum > MANIFEST.sha256 )
  local before after out rc
  before="$(tree_hash)"
  out="$(bash "$INSTALL" "$dir" 2>&1)"; rc=$?
  after="$(tree_hash)"
  local last; last="$(grep -o '^\[[0-9]/8\]' <<<"$out" | tail -1)"
  printf '\n=== %s\n' "$name"
  grep -E "FAIL|refus|cannot|history_length|joint order" <<<"$out" | grep -v "^\s*ok" | head -4 | sed 's/^/    /'
  if [[ $rc -ne 0 && "$last" == "[1/8]" && "$before" == "$after" ]] && grep -qi -- "$want" <<<"$out"; then
    echo "  ok: refused at step 1 for '$want'; deploy tree unchanged ($after)"
  else
    echo "  NOT OK: exit $rc, last step $last, tree $before -> $after, reason '$want' $(grep -qi -- "$want" <<<"$out" && echo found || echo MISSING)"
    bad=$((bad+1))
  fi
}

echo "deploy tree $R1_DEPLOY_ROOT, hash before: $(tree_hash)"
case_install "1. history_length 5 -> 6" "history_length" "
import json,pathlib
p=pathlib.Path('policy_interface.json'); d=json.loads(p.read_text())
d['observation']['history_length']=6
p.write_text(json.dumps(d,indent=2))"
case_install "2. actuator_gains.json missing one joint" "joint order" "
import json,pathlib
p=pathlib.Path('actuator_gains.json'); d=json.loads(p.read_text())
for k in ('joint_names','group','stiffness','damping','effort_limit'):
    d[k]=d[k][:-1]
p.write_text(json.dumps(d,indent=2))"
case_install "3. observation term the bridge cannot compute: base_lin_vel" "base_lin_vel" "
import json,pathlib
p=pathlib.Path('policy_interface.json'); d=json.loads(p.read_text())
o=d['observation']
o['terms'].insert(0,{'name':'base_lin_vel','width':3})
o['frame_dim']+=3; o['total_dim']=o['frame_dim']*o['history_length']
p.write_text(json.dumps(d,indent=2))"
echo
echo "-------------------------------------------"
echo "negative installs: $((3-bad))/3 refused before anything was written"
exit $bad
