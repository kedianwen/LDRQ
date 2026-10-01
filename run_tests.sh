#!/usr/bin/env bash
# Every check that needs no GPU, no ROS and no robot: Python 3.8+ standard
# library and coreutils only. The same script runs in CI
# (.github/workflows/tests.yml), so the test counts it prints are the only
# ones to trust -- the documentation deliberately does not copy them.
#
#   ./run_tests.sh        # summary only; exits non-zero if anything fails
#   ./run_tests.sh -v     # also print each suite's full output
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY="${PYTHON:-python3}"
VERBOSE=0; [[ "${1:-}" == "-v" ]] && VERBOSE=1
failed=0

run() {  # run <name> <command...>; prints the command's last line
  local name="$1"; shift
  local out rc
  out="$("$@" 2>&1)"; rc=$?
  [[ $VERBOSE -eq 1 ]] && printf '%s\n' "$out"
  printf '%-34s %-6s %s\n' "$name" "$([[ $rc -eq 0 ]] && echo ok || echo FAIL)" "$(printf '%s\n' "$out" | tail -1)"
  if [[ $rc -ne 0 ]]; then
    failed=1
    [[ $VERBOSE -eq 0 ]] && printf '%s\n' "$out" | tail -20 | sed 's/^/    | /'
  fi
}

echo "python: $("$PY" --version 2>&1)"
run "mission_ctl core"            "$PY" mission_ctl/tests/test_core.py
run "mission_ctl English (ask)"   "$PY" mission_ctl/tests/test_nl.py
run "eval sets vs shipped config" "$PY" mission_ctl/eval/run_nl_eval.py --check --set all
run "llm coexist selftest"        "$PY" llm/coexist.py --selftest
run "policy_pack refusals"        env PATH="$(dirname "$(command -v "$PY")"):$PATH" bash policy_pack/tests/run_tests.sh
run "deployed policy SHA256"      bash -c 'cd models/week04_nohead && sha256sum --quiet -c SHA256SUMS && echo "$(wc -l < SHA256SUMS) files match SHA256SUMS"'
run "mission dry run"             bash -c '"$0" mission_ctl/r1_mission_cli.py run "walk 3s@0.3; turn left 90" --dry-run | grep -o "\"final_state\": \"DONE\""' "$PY"

exit $failed
