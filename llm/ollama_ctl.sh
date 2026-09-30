#!/usr/bin/env bash
# The robot-local model server behind `r1_mission_cli.py ask` (stage C).
#
# User space: no sudo, no systemd, no apt. The runtime (ollama/) and the models
# (models/) sit next to this script, and the server listens on 127.0.0.1 only.
#
#   bash ollama_ctl.sh start       start the server and wait until it answers
#   bash ollama_ctl.sh warm        load the model now (the first load is the slow one)
#   bash ollama_ctl.sh health      one real `ask` request within a deadline; exit 0 = healthy
#   bash ollama_ctl.sh status      pid, version, which GPU/library, loaded models
#   bash ollama_ctl.sh stop | restart
#   bash ollama_ctl.sh watchdog    foreground loop: health every 30 s, restart after 2 fails
#   bash ollama_ctl.sh models      installed models
#   bash ollama_ctl.sh log         follow the server log
#
# Why the health check GENERATES instead of pinging: on the Orin NX, llama-server
# (Ollama's runner) sometimes comes up and never serves a request (llama.cpp issue
# #29499, still open 2026-09-30). The Ollama process keeps answering /api/version
# the whole time, so only a generation shows the fault. A restart clears it.
#
# Why the health check is `r1_mission_cli.py ask-check --once`, not a bare generate:
# Ollama reloads the model whenever a request asks for another placement, and the
# server keeps only the last prompt in its cache. A bare generate with default
# options put the model back on the GPU after mission.yaml had moved it to the CPU,
# and pushed the ~1000-token `ask` prompt out of the cache (about 1.6 s to re-read
# on the GPU, several times that on the CPU). ask-check sends exactly what `ask`
# sends: mission.yaml's model and placement, and the real prompt.
#
# Model: mission.yaml llm_model; R1_LLM_MODEL overrides it.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI="${R1_MISSION_CLI:-$HERE/../mission_ctl/r1_mission_cli.py}"
CFG="$(dirname "$CLI")/config/mission.yaml"
cfg() {  # cfg KEY -> its value in mission.yaml (flat `key: value`), empty if absent
    [ -f "$CFG" ] && sed -n "s/^$1:[[:space:]]*\([^#[:space:]]*\).*/\1/p" "$CFG" | head -1
}
PREFIX="${R1_OLLAMA_PREFIX:-$HERE/ollama}"
BIN="$PREFIX/bin/ollama"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HERE/models}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NOPRUNE=1          # never delete a blob that was copied in by hand
export OLLAMA_NO_CLOUD=1
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-2048}"
# Jetson: select the JetPack 5 CUDA runner explicitly instead of relying on the
# /etc/nv_tegra_release parse, and admit the integrated GPU explicitly (Ollama
# drops integrated GPUs of some kinds by default; CUDA ones are allowed today).
export JETSON_JETPACK="${JETSON_JETPACK:-5.1.1}"
export OLLAMA_IGPU_ENABLE=1
MODEL="${R1_LLM_MODEL:-$(cfg llm_model)}"
MODEL="${MODEL:-qwen3:1.7b}"
LOG="$HERE/ollama.log"
PIDF="$HERE/ollama.pid"
URL="http://$OLLAMA_HOST"

# HTTP through python3 (always present: mission_ctl needs it), not curl.
http() {  # http METHOD PATH TIMEOUT [JSON_BODY]  -> prints the body, exit 0 on 2xx
    python3 - "$1" "$URL$2" "$3" "${4:-}" <<'PY'
import sys, urllib.request
method, url, timeout, body = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
req = urllib.request.Request(url, data=body.encode() if body else None, method=method,
                             headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=timeout) as r:
        sys.stdout.write(r.read().decode("utf-8", "replace"))
except Exception as exc:  # noqa
    sys.stderr.write("{}\n".format(exc))
    sys.exit(1)
PY
}

pid_alive() {
    [ -f "$PIDF" ] || return 1
    local pid; pid="$(cat "$PIDF")"
    kill -0 "$pid" 2>/dev/null && grep -q ollama "/proc/$pid/cmdline" 2>/dev/null
}

version() { http GET /api/version 3 >/dev/null 2>&1; }

start() {
    if pid_alive; then echo "already running (pid $(cat "$PIDF"))"; return 0; fi
    if version; then
        echo "something else already serves $URL -- stop it first (pgrep -af ollama)"
        return 1
    fi
    if [ ! -x "$BIN" ]; then
        echo "no runtime at $BIN -- extract the stage C LLM runtime package first"
        return 2
    fi
    echo "starting: $BIN serve on $OLLAMA_HOST, models in $OLLAMA_MODELS"
    echo "===== start $(date -Iseconds) =====" >>"$LOG"
    # setsid: a Ctrl-C in this terminal later must not reach the server.
    setsid nohup "$BIN" serve >>"$LOG" 2>&1 < /dev/null &
    echo $! >"$PIDF"
    for _ in $(seq 1 30); do
        if version; then
            echo "up (pid $(cat "$PIDF")). GPU discovery:"
            grep -E "inference compute|dropping integrated|jetpack|no compatible GPUs" "$LOG" \
                | tail -3 | sed 's/^/  /'
            return 0
        fi
        sleep 1
    done
    echo "the server did not answer within 30 s; see $LOG"
    return 1
}

stop() {
    if pid_alive; then
        local pid; pid="$(cat "$PIDF")"
        kill "$pid" 2>/dev/null
        for _ in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -9 "$pid" 2>/dev/null
    fi
    rm -f "$PIDF"
    # A runner left behind by a killed server still holds the model's memory.
    pkill -f "$PREFIX/lib/ollama/llama-server" 2>/dev/null
    pkill -f "$BIN runner" 2>/dev/null
    echo "stopped"
}

since() { awk -v a="$1" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b - a}'; }

health() {  # health [deadline_s]
    local deadline="${1:-60}"
    if ! version; then echo "health: FAIL -- nothing answers at $URL"; return 1; fi
    local t0; t0=$(date +%s.%N)
    if [ -f "$CLI" ]; then
        local out rc
        out="$(python3 "$CLI" ask-check --once --llm-url "$URL" --llm-timeout "$deadline" \
               ${R1_LLM_MODEL:+--model "$R1_LLM_MODEL"} 2>&1)"
        rc=$?
        case "$rc" in
            0) echo "health: OK -- $MODEL answered in $(since "$t0") s, and the plan is right"
               echo "$out" | grep -E "^model |^request" | sed 's/^/  /'
               return 0 ;;
            1) echo "health: OK -- $MODEL answered in $(since "$t0") s, but WRONG (the server works)"
               echo "$out" | grep -E "^model |^request|^WRONG" | sed 's/^/  /'
               return 0 ;;
            3) echo "health: FAIL -- $MODEL did not answer within ${deadline} s"
               echo "$out" | tail -4 | sed 's/^/  /'
               return 1 ;;
            *) echo "health: FAIL -- ask-check could not run (exit $rc), so the server was not asked"
               echo "$out" | tail -4 | sed 's/^/  /'
               return 1 ;;
        esac
    fi
    # No mission_ctl next to llm/: a bare 1-token generation, Ollama's default placement.
    local body
    body="{\"model\":\"$MODEL\",\"prompt\":\"ok\",\"stream\":false,\"options\":{\"num_predict\":1},\"keep_alive\":\"$OLLAMA_KEEP_ALIVE\"}"
    if http POST /api/generate "$deadline" "$body" >/dev/null 2>"$HERE/.health.err"; then
        echo "health: OK -- $MODEL answered in $(since "$t0") s (bare generate: no $CLI)"
        return 0
    fi
    echo "health: FAIL -- $MODEL did not answer within ${deadline} s ($(head -c 200 "$HERE/.health.err"))"
    return 1
}

status() {
    if pid_alive; then echo "server: pid $(cat "$PIDF")"; else echo "server: not running (from this script)"; fi
    echo "mission.yaml: llm_model ${MODEL}, llm_num_gpu $(cfg llm_num_gpu) (-1 = Ollama's default, the GPU; 0 = the CPU), llm_num_thread $(cfg llm_num_thread)"
    if version; then
        echo "version: $(http GET /api/version 3)"
        echo "loaded models (size_vram > 0 = on the GPU):"
        http GET /api/ps 3 | python3 -c '
import json, sys
d = json.load(sys.stdin)
for m in d.get("models", []):
    print("  {:<16} size {:6.0f} MB  on GPU {:6.0f} MB  until {}".format(
        m["name"], m["size"] / 1e6, m.get("size_vram", 0) / 1e6, m.get("expires_at", "")[:19]))
if not d.get("models"):
    print("  (none)")'
    else
        echo "version: nothing answers at $URL"
    fi
    echo "GPU discovery (last start):"
    grep -E "inference compute|dropping integrated|jetpack|no compatible GPUs" "$LOG" 2>/dev/null \
        | tail -3 | sed 's/^/  /'
}

watchdog() {
    local fails=0
    echo "watchdog: health every 30 s, restart after 2 failures (Ctrl-C to stop)"
    while true; do
        if health 60 >/dev/null; then
            fails=0
        else
            fails=$((fails + 1))
            echo "$(date -Iseconds) health failed ($fails)" | tee -a "$HERE/watchdog.log"
            if [ "$fails" -ge 2 ]; then
                echo "$(date -Iseconds) restarting" | tee -a "$HERE/watchdog.log"
                stop >/dev/null; start >/dev/null && health 180 | tee -a "$HERE/watchdog.log"
                fails=0
            fi
        fi
        sleep 30
    done
}

case "${1:-status}" in
    start) start ;;
    stop) stop ;;
    restart) stop; start ;;
    status) status ;;
    health) health "${2:-60}" ;;
    warm) echo "loading $MODEL (the first load reads it from disk)..."; health "${2:-300}" ;;
    watchdog) watchdog ;;
    models) OLLAMA_HOST="$OLLAMA_HOST" "$BIN" list ;;
    log) tail -n 50 -f "$LOG" ;;
    *) sed -n '2,29p' "$0"; exit 2 ;;
esac
