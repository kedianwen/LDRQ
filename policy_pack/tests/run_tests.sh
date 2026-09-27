#!/usr/bin/env bash
# What verify_bundle.py REFUSES. A validator is only worth the bad bundles it
# stops, so every case below is a real way a policy swap goes wrong, and each
# asserts on the REASON text, not just on a non-zero exit -- a validator that
# fails for the wrong reason sends the operator down the wrong path.
#
#   ./run_tests.sh            # build a scratch bundle and mutate it
#   ./run_tests.sh <bundle>   # mutate a copy of an existing bundle
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACK="$(dirname "$HERE")"
VERIFY="$PACK/verify_bundle.py"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if [[ $# -ge 1 ]]; then
  SRC="$1"
else
  python3 "$PACK/make_bundle.py" --out "$WORK/src" >/dev/null 2>&1 || {
    echo "cannot build a scratch bundle (need deploy/interface + artifacts)"; exit 2; }
  SRC="$(find "$WORK/src" -mindepth 1 -maxdepth 1 -type d | head -1)"
fi
echo "source bundle: $SRC"
echo

pass=0 fail=0

# Mutate a fresh copy, re-hash the manifest (so we test the check we mean to
# test and not the manifest check every time), then expect a specific refusal.
case_expect() {
  local name="$1" want="$2"; shift 2
  local dir="$WORK/case"; rm -rf "$dir"; cp -r "$SRC" "$dir"
  ( cd "$dir" && "$@" ) || { echo "  SETUP FAILED: $name"; fail=$((fail+1)); return; }
  if [[ "${REHASH:-1}" == "1" ]]; then
    ( cd "$dir" && find . -type f ! -name MANIFEST.sha256 -printf '%P\n' | sort \
        | xargs sha256sum > MANIFEST.sha256 )
  fi
  local out rc
  out="$("$VERIFY" "$dir" 2>&1)"; rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "  FAIL  $name -- bundle was ACCEPTED, expected refusal"
    fail=$((fail+1)); return
  fi
  if grep -qi -- "$want" <<<"$out"; then
    echo "  ok    $name"
    echo "          -> $(grep -i 'VERDICT: FAIL' <<<"$out" | head -1 | cut -c1-96)"
    pass=$((pass+1))
  else
    echo "  FAIL  $name -- refused, but not for '$want':"
    sed 's/^/          /' <<<"$out" | tail -4
    fail=$((fail+1))
  fi
}

jset() { python3 - "$@" <<'PY'
import json, sys, pathlib
path, expr = pathlib.Path(sys.argv[1]), sys.argv[2]
d = json.loads(path.read_text())
exec(expr, {"d": d})
path.write_text(json.dumps(d, indent=2) + "\n")
PY
}
export -f jset 2>/dev/null || true

echo "--- the bundle as built must pass ---"
if "$VERIFY" "$SRC" --quiet; then echo "  ok    unmodified bundle accepted"; pass=$((pass+1));
else echo "  FAIL  unmodified bundle rejected"; fail=$((fail+1)); fi
echo
echo "--- refusals ---"

# A history length that disagrees with total_dim. This is the one that would
# otherwise be caught only at engine load, after a 25 s build.
case_expect "history_length 5 -> 6, total_dim untouched" "history_length" \
  python3 -c "
import json,pathlib
p=pathlib.Path('policy_interface.json'); d=json.loads(p.read_text())
d['observation']['history_length']=6
p.write_text(json.dumps(d,indent=2))"

# An observation term the C++ bridge has no code to compute. The failure mode
# without this check is a 50 Hz zero-filled term and a week of sim2real hunting.
case_expect "unknown obs term base_lin_vel" "cannot assemble" \
  python3 -c "
import json,pathlib
p=pathlib.Path('policy_interface.json'); d=json.loads(p.read_text())
o=d['observation']
o['terms'].insert(0,{'name':'base_lin_vel','width':3})
o['frame_dim']+=3; o['total_dim']=o['frame_dim']*o['history_length']
p.write_text(json.dumps(d,indent=2))"

# Same six terms, wrong order. Dimensions all still add up.
case_expect "obs terms reordered, dims still consistent" "out of order" \
  python3 -c "
import json,pathlib
p=pathlib.Path('policy_interface.json'); d=json.loads(p.read_text())
t=d['observation']['terms']; t[0],t[1]=t[1],t[0]
p.write_text(json.dumps(d,indent=2))"

# Gains for 25 joints instead of 26.
case_expect "actuator_gains missing one joint" "joint order" \
  python3 -c "
import json,pathlib
p=pathlib.Path('actuator_gains.json'); d=json.loads(p.read_text())
for k in ('joint_names','group','stiffness','damping','effort_limit'):
    d[k]=d[k][:-1]
p.write_text(json.dumps(d,indent=2))"

# Gains in a plausible but different order -- alphabetical. Every array is the
# right length; the bridge would apply the ankle's kp to a hip.
case_expect "actuator_gains sorted alphabetically" "joint order" \
  python3 -c "
import json,pathlib
p=pathlib.Path('actuator_gains.json'); d=json.loads(p.read_text())
order=sorted(range(len(d['joint_names'])), key=lambda i: d['joint_names'][i])
for k in ('joint_names','group','stiffness','damping','effort_limit'):
    d[k]=[d[k][i] for i in order]
p.write_text(json.dumps(d,indent=2))"

# kd = 0 on a leg: this is the W06 defect (undamped legs) in file form.
case_expect "damping 0 on a leg joint" "damping is <= 0" \
  python3 -c "
import json,pathlib
p=pathlib.Path('actuator_gains.json'); d=json.loads(p.read_text())
d['damping'][0]=0.0
p.write_text(json.dumps(d,indent=2))"

# An action index that names one joint and points at another.
case_expect "action index points at the wrong joint" "but articulation" \
  python3 -c "
import json,pathlib
p=pathlib.Path('policy_interface.json'); d=json.loads(p.read_text())
d['action']['joint_ids_in_articulation'][0]=3
p.write_text(json.dumps(d,indent=2))"

# Envelope with lo > hi.
case_expect "envelope vx lo > hi" "lo .* > hi" \
  python3 -c "
import json,pathlib
p=pathlib.Path('command_envelope.json'); d=json.loads(p.read_text())
d['vx']=[1.0,0.0]
p.write_text(json.dumps(d,indent=2))"

# A hand-edited json with the manifest left alone -- the most likely real
# accident. REHASH=0 keeps the stale manifest.
REHASH=0 case_expect "json edited, manifest not regenerated" "MANIFEST" \
  python3 -c "
import json,pathlib
p=pathlib.Path('command_envelope.json'); d=json.loads(p.read_text())
d['wz']=[-2.0,2.0]
p.write_text(json.dumps(d,indent=2))"

# An extra file smuggled in, manifest regenerated around the listed ones only.
REHASH=0 case_expect "unlisted extra file in the bundle" "not in MANIFEST" \
  bash -c "echo 'kp_scale: 0.5' > tuning_override.yaml"

# The ONNX swapped for a different one, provenance untouched.
REHASH=1 case_expect "policy.onnx replaced, provenance stale" "onnx_sha256" \
  bash -c "head -c 363847 /dev/zero > policy.onnx"

echo
echo "-------------------------------------------"
printf "pass %d  fail %d\n" "$pass" "$fail"
[[ $fail -eq 0 ]] || exit 1
