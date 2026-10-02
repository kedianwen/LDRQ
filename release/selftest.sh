#!/usr/bin/env bash
# Checks for a freshly extracted release, on the robot. Moves nothing, starts
# nothing, needs no ROS stack. Run from anywhere:   bash selftest.sh
# Lines marked "warn" are about the robot's system and do not fail the test.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
fail=0
ok()   { printf '  ok    %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; fail=$((fail+1)); }
warn() { printf '  warn  %s\n' "$*"; }
run()  { local name="$1"; shift; local out; out="$("$@" 2>&1)" && ok "$name: $(tail -1 <<<"$out")" \
         || { bad "$name"; tail -8 <<<"$out" | sed 's/^/        | /'; }; }

echo "r1-robot $(cat VERSION 2>/dev/null) in $PWD"
echo "-- files"
run "release files match FILES.sha256" bash -c 'sha256sum --quiet -c FILES.sha256 && echo "$(wc -l < FILES.sha256) files"'
if [[ -f llm/models/MODEL.sha256 ]]; then
  run "model files match MODEL.sha256" bash -c 'cd llm/models && sha256sum --quiet -c MODEL.sha256 && echo "$(wc -l < MODEL.sha256) files"'
else
  bad "no model: extract qwen3-1.7b_r1-robot-*.tar.gz in the same directory as this release (it adds llm/models/)"
fi
[[ -x llm/ollama/bin/ollama ]] && ok "ollama runtime present" || bad "llm/ollama/bin/ollama missing or not executable (copied as zip?)"

echo "-- code (Python $(python3 -c 'import platform;print(platform.python_version())'))"
B="$(ls -d bundles/*/ | head -1)"
run "mission_ctl core tests"        python3 mission_ctl/tests/test_core.py
run "mission_ctl English tests"     python3 mission_ctl/tests/test_nl.py
run "eval sets vs shipped config"   python3 mission_ctl/eval/run_nl_eval.py --check --set all
run "coexist selftest"              python3 llm/coexist.py --selftest
run "policy bundle valid"           python3 policy_pack/verify_bundle.py "$B"
run "bad bundles refused"           bash policy_pack/tests/run_tests.sh "$B"

echo "-- robot system (needed for install_bundle.sh and the stack)"
grep -q "R35" /etc/nv_tegra_release 2>/dev/null && ok "JetPack 5 ($(head -1 /etc/nv_tegra_release | cut -c1-40))" \
  || warn "not JetPack 5 (/etc/nv_tegra_release): fine for these tests, not for the stack"
[[ -f /opt/ros/foxy/setup.bash ]] && ok "ROS 2 foxy" || warn "no /opt/ros/foxy"
[[ -d /usr/local/include/unitree ]] && ok "unitree_sdk2 headers in /usr/local" \
  || warn "no /usr/local/include/unitree: the bridge cannot be built (install unitree_sdk2)"
ls /usr/lib/aarch64-linux-gnu/libnvinfer.so.8* >/dev/null 2>&1 && ok "TensorRT 8 libraries" \
  || warn "no TensorRT 8 in /usr/lib/aarch64-linux-gnu"

echo
[[ $fail -eq 0 ]] && echo "selftest: OK" || echo "selftest: $fail FAILED"
exit $fail
