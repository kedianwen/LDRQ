#!/usr/bin/env bash
# Stage C 3.1 on the robot: what there is for a local model, and whether a cloud
# fallback exists at all. Reads only; changes nothing; no sudo.
#
#   bash env_check.sh | tee ~/orin_commissioning/stageC_env_$(date +%H%M%S).txt
#
# The network part answers one question: is there ANY route to a hosted model?
# All three tests must pass for a cloud fallback to exist. If they fail, the model
# runs on the robot or on a machine on the robot's own network (--llm-url) -- which
# is the design anyway: nothing at run time needs the internet.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "== when: $(date -Iseconds)  host: $(hostname)"

echo; echo "== memory"
free -h
echo; echo "== disk under ~/kdw_deploy"
df -h "$HOME/kdw_deploy" 2>/dev/null | tail -1

echo; echo "== CPU and Jetson"
echo "cores: $(nproc)"
head -1 /etc/nv_tegra_release 2>/dev/null || echo "(no /etc/nv_tegra_release)"
if command -v nvpmodel >/dev/null; then nvpmodel -q 2>/dev/null | head -2; fi
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
    [ -r "$f" ] && printf "%s " "$(( $(cat "$f") / 1000 ))"
done; echo "MHz (current CPU clocks; locked clocks show the same max on every core)"
for f in /sys/class/devfreq/*ga10b*/cur_freq; do
    [ -r "$f" ] && echo "GPU clock: $(( $(cat "$f") / 1000000 )) MHz"
done

echo; echo "== network 1/3: interfaces and routes"
ip -brief addr 2>/dev/null || ip addr
ip route 2>/dev/null
if ip route 2>/dev/null | grep -q '^default'; then echo "default route: yes"; else echo "default route: NONE"; fi

echo; echo "== network 2/3: DNS"
for h in ollama.com huggingface.co api.openai.com; do
    if getent hosts "$h" >/dev/null 2>&1; then echo "  $h -> $(getent hosts "$h" | head -1 | awk '{print $1}')"
    else echo "  $h: no answer"; fi
done

echo; echo "== network 3/3: HTTPS round trip (5 s timeout each)"
python3 - <<'PY'
import time, urllib.request
for url in ("https://ollama.com", "https://api.openai.com/v1/models"):
    t0 = time.time()
    try:
        urllib.request.urlopen(url, timeout=5)
        print("  {}: reachable in {:.0f} ms".format(url, (time.time() - t0) * 1000))
    except Exception as exc:  # an HTTP 401 from api.openai.com still proves the route
        code = getattr(exc, "code", None)
        if code:
            print("  {}: reachable (HTTP {}) in {:.0f} ms".format(url, code, (time.time() - t0) * 1000))
        else:
            print("  {}: NOT reachable ({})".format(url, str(exc)[:80]))
PY

echo; echo "== local model server"
bash "$HERE/ollama_ctl.sh" status
