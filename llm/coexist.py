#!/usr/bin/env python3
"""Does the 50 Hz control loop keep its timing while a local model decodes on the
same Orin NX? (stage C, plan 3.6)

Runs load phases against the model server -- idle, continuous GPU decode, the real
`ask` request back to back, continuous CPU decode, idle again -- while the stack is
running, then reads the stack's OWN timing lines out of its log for each phase:

  bridge  "RUNNING | obs 50.0 Hz ... | setpoint lag p50=0.6 p95=1.1 max=2.0 ms"  (every 5 s)
          "observation rate 44.1 Hz below 45.0 Hz (1/3)"                        (any 1 s window)
          "DEGRADED: ..."
  policy  "250 cycles in 5.0s (50.0 Hz) | infer us p50=455 p95=... p99=... max=..."

Those are the numbers the stack already reports, so nothing new runs inside the
control loop. Each LLM phase is judged against the first idle one, in two tiers:

  limits (what the control loop cannot give up; exit status 1 if any is broken)
  * obs >= 45 Hz in every sample, no "below" window, no DEGRADED
  * no failed inference, and the slowest inference inside the control period
  * setpoint lag p95 under one control step (training drew the lag from 0-1 steps)

  targets (plan 3.6 as written)
  * policy inference p99 <= 10 % of the control period (2 ms at 50 Hz)
  * setpoint lag p95 no more than 1 ms worse than idle

On the robot (2026-09-30) the model on the GPU kept the limits and missed both targets:
while it decodes, the policy's TensorRT inference waits for the GPU, up to about 5 ms.
It stays on the GPU by decision; the targets are what GPU-side optimization aims at.
Four CPU threads met both tiers.

Robot, stack up in terminal 1 with its output saved:
  PYTHONUNBUFFERED=1 ros2 launch r1_hw_bridge r1_stack.launch.py ... 2>&1 | tee ~/orin_commissioning/stack_$(date +%H%M%S).log
Terminal 2:
  python3 coexist.py --stack-log ~/orin_commissioning/stack_XXXXXX.log
  python3 coexist.py --report ~/orin_commissioning/coexist_XXXXXX.json   # re-print
  python3 coexist.py --selftest                                          # no robot, no model

Stdlib only, Python 3.8.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
LAG_WINDOW_S = 11.0     # the bridge's lag stats cover its last 512 samples (~10.2 s)
POLICY_PERIOD_S = 5.0   # the policy node's stats_period_s

TS = r"\[(\d{9,}\.\d+)\]"
RX_BRIDGE = re.compile(
    TS + r" \[r1_hw_bridge\]: (\w+) \| obs ([\d.]+) Hz \(want [\d.]+\) \| cmd ([\d.]+) Hz "
    r"\(want [\d.]+\) \| output=(\w+) \| kp_scale=([\d.]+) \| crc_fail=(\d+) \| "
    r"setpoint lag p50=([\d.]+) p95=([\d.]+) max=([\d.]+) ms")
RX_POLICY = re.compile(
    TS + r" \[r1_policy_node\]: (\d+) cycles in ([\d.]+)s \(([\d.]+) Hz\) \| infer us "
    r"p50=(\d+) p95=(\d+) p99=(\d+) max=(\d+) \| budget (\d+) us \| failures=(\d+)")
RX_SLOW = re.compile(TS + r" \[r1_hw_bridge\]: observation rate ([\d.]+) Hz below")
RX_DEGR = re.compile(TS + r" \[r1_hw_bridge\]: DEGRADED: (.*)")
RX_LAGW = re.compile(TS + r" \[r1_hw_bridge\]: setpoint lag p95=([\d.]+) ms exceeds")

LOAD_PROMPT = ("Write the numbers from 1 to 400 in words, separated by commas, "
               "and nothing else.")


# ------------------------------------------------------------------ parsing
def parse_log(lines):
    ev = {"bridge": [], "policy": [], "slow": [], "degraded": [], "lagwarn": []}
    for line in lines:
        m = RX_BRIDGE.search(line)
        if m:
            ev["bridge"].append({"t": float(m.group(1)), "state": m.group(2),
                                 "obs_hz": float(m.group(3)), "cmd_hz": float(m.group(4)),
                                 "output": m.group(5), "kp_scale": float(m.group(6)),
                                 "lag_p50": float(m.group(8)), "lag_p95": float(m.group(9)),
                                 "lag_max": float(m.group(10))})
            continue
        m = RX_POLICY.search(line)
        if m:
            ev["policy"].append({"t": float(m.group(1)), "cycles": int(m.group(2)),
                                 "hz": float(m.group(4)), "p50": int(m.group(5)),
                                 "p95": int(m.group(6)), "p99": int(m.group(7)),
                                 "max": int(m.group(8)), "budget": int(m.group(9)),
                                 "failures": int(m.group(10))})
            continue
        for key, rx in (("slow", RX_SLOW), ("degraded", RX_DEGR), ("lagwarn", RX_LAGW)):
            m = rx.search(line)
            if m:
                ev[key].append({"t": float(m.group(1)), "what": m.group(2)})
                break
    return ev


def summarize_phase(ev, start, end):
    b = [x for x in ev["bridge"] if start + LAG_WINDOW_S <= x["t"] <= end + 0.5]
    p = [x for x in ev["policy"] if x["t"] - POLICY_PERIOD_S >= start - 0.5
         and x["t"] <= end + 0.5]

    def inside(key):
        return [x for x in ev[key] if start <= x["t"] <= end + 0.5]

    out = {"bridge_samples": len(b), "policy_samples": len(p),
           "slow_windows": len(inside("slow")), "degraded": len(inside("degraded")),
           "lag_warnings": len(inside("lagwarn"))}
    if b:
        out.update({"obs_hz_min": min(x["obs_hz"] for x in b),
                    "obs_hz_mean": round(statistics.mean(x["obs_hz"] for x in b), 2),
                    "cmd_hz_min": min(x["cmd_hz"] for x in b),
                    "lag_p50_median": statistics.median(x["lag_p50"] for x in b),
                    "lag_p95_max": max(x["lag_p95"] for x in b),
                    "lag_max_max": max(x["lag_max"] for x in b),
                    "states": sorted(set(x["state"] for x in b)),
                    "output": sorted(set(x["output"] for x in b))})
    if p:
        out.update({"policy_hz_min": min(x["hz"] for x in p),
                    "infer_p50_median": statistics.median(x["p50"] for x in p),
                    "infer_p95_max": max(x["p95"] for x in p),
                    "infer_p99_max": max(x["p99"] for x in p),
                    "infer_max_max": max(x["max"] for x in p),
                    "budget_us": p[-1]["budget"],
                    "failures": p[-1]["failures"] - p[0]["failures"]})
    return out


LIMIT, TARGET = "limit", "target"


def verdict(phase, idle):
    """The criteria for one phase, against the idle baseline, as (tier, name, ok,
    detail). Limits are what the control loop cannot give up; targets are plan 3.6 as
    written (see the module docstring)."""
    if not phase.get("bridge_samples") or not phase.get("policy_samples"):
        return [(LIMIT, "enough samples", False,
                 "bridge {} / policy {} lines in the window".format(
                     phase.get("bridge_samples"), phase.get("policy_samples")))]
    budget = phase.get("budget_us", 20000)
    step_ms = budget / 1000.0
    base = idle.get("lag_p95_max") if idle else None
    return [
        (LIMIT, "obs >= 45 Hz, no slow window, no DEGRADED",
         phase["obs_hz_min"] >= 45.0 and phase["slow_windows"] == 0 and phase["degraded"] == 0,
         "min {:.1f} Hz, {} slow windows, {} degraded".format(
             phase["obs_hz_min"], phase["slow_windows"], phase["degraded"])),
        (LIMIT, "no failed inference, max < {:.0f} us".format(budget),
         phase.get("failures", 0) == 0 and phase["infer_max_max"] < budget,
         "{} failed, max {} us".format(phase.get("failures", 0), phase["infer_max_max"])),
        (LIMIT, "lag p95 < one step ({:.0f} ms)".format(step_ms),
         phase["lag_p95_max"] < step_ms, "p95 max {:.1f} ms".format(phase["lag_p95_max"])),
        (TARGET, "infer p99 <= {:.0f} us (plan 3.6)".format(0.1 * budget),
         phase["infer_p99_max"] <= 0.1 * budget, "p99 max {} us".format(phase["infer_p99_max"])),
        (TARGET, "lag p95 <= idle + 1 ms (plan 3.6)",
         base is None or phase["lag_p95_max"] <= base + 1.0,
         "p95 max {:.1f} ms (idle {})".format(
             phase["lag_p95_max"], "{:.1f} ms".format(base) if base is not None else "n/a")),
    ]


def tiers_ok(phase, idle):
    """(limits kept, targets met) for one phase."""
    v = verdict(phase, idle)
    return (all(ok for t, _, ok, _ in v if t == LIMIT),
            all(ok for t, _, ok, _ in v if t == TARGET))


# ---------------------------------------------------------------- load
def _post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get(url, timeout=3.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def phase_options(name, cpu_threads):
    if name == "gpu":
        return {"num_gpu": 999}
    if name == "cpu":
        o = {"num_gpu": 0}
        if cpu_threads:
            o["num_thread"] = cpu_threads
        return o
    return None


def run_generate_load(url, model, options, until, stats):
    body = {"model": model, "prompt": LOAD_PROMPT, "stream": False, "keep_alive": "30m",
            "options": dict(options, num_predict=200, temperature=0.7)}
    while time.time() < until:
        try:
            d = _post(url + "/api/generate", body, timeout=120)
            stats["requests"] += 1
            stats["tokens"] += d.get("eval_count") or 0
            stats["gen_s"] += (d.get("eval_duration") or 0) / 1e9
        except Exception as exc:  # noqa: BLE001 -- counted, not fatal
            stats["errors"] += 1
            stats["last_error"] = str(exc)[:200]
            time.sleep(1.0)


ASK_TEXTS = ["Walk forward 2 meters, then turn left.", "Turn right 45 degrees.",
             "Walk forward for 5 seconds, wait 2 seconds, then turn around.",
             "Strafe left 2 meters.", "Walk forward 10 feet."]


def _mission():
    sys.path.insert(0, str(HERE.parent / "mission_ctl"))
    from r1_mission import nl, node as N
    return nl, N


def configured_ask_options():
    """The placement `ask` itself uses: mission.yaml's llm_num_gpu / llm_num_thread."""
    _, N = _mission()
    return N.llm_options(N.read_flat_yaml(str(N.DEFAULT_CFG)))


def warm_ask(url, model, options):
    """One real `ask` request: loads the model where `ask` would put it and leaves the
    `ask` prompt in the server's cache, as every later request finds it."""
    nl, N = _mission()
    lim = N.limits_from_cfg(N.read_flat_yaml(str(N.DEFAULT_CFG)))
    be = nl.make_backend("ollama", url, model, 300, options)
    be.transcribe(nl.normalize(ASK_TEXTS[0])[0], nl.transcription_schema(lim))


def run_ask_load(url, model, until, stats, options=None):
    """The real request: same prompt, schema and options as `ask`, back to back."""
    nl, N = _mission()
    lim = N.limits_from_cfg(N.read_flat_yaml(str(N.DEFAULT_CFG)))
    be = nl.make_backend("ollama", url, model, 120, options)
    texts = ASK_TEXTS
    i = 0
    while time.time() < until:
        try:
            _, meta = be.transcribe(nl.normalize(texts[i % len(texts)])[0],
                                    nl.transcription_schema(lim))
            stats["requests"] += 1
            stats["tokens"] += meta.get("gen_tokens") or 0
            stats["gen_s"] += meta.get("gen_s") or 0.0
            stats.setdefault("wall_s", []).append(meta.get("wall_s"))
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            stats["last_error"] = str(exc)[:200]
            time.sleep(1.0)
        i += 1


class Tegrastats(object):
    """Optional: RAM and GPU load per phase. Lines are stamped on arrival."""

    def __init__(self):
        self.lines, self.proc = [], None
        exe = shutil.which("tegrastats")
        if not exe:
            return
        try:
            self.proc = subprocess.Popen([exe, "--interval", "1000"], stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, universal_newlines=True)
        except OSError:
            self.proc = None
            return
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.append((time.time(), line.strip()))

    def stop(self):
        if self.proc:
            self.proc.terminate()

    def summarize(self, start, end):
        ram, gpu, cpu = [], [], []
        for t, line in self.lines:
            if not (start <= t <= end):
                continue
            m = re.search(r"RAM (\d+)/(\d+)MB", line)
            if m:
                ram.append(int(m.group(1)))
            m = re.search(r"GR3D_FREQ (\d+)%", line)
            if m:
                gpu.append(int(m.group(1)))
            m = re.search(r"CPU \[([^\]]*)\]", line)
            if m:
                loads = [int(x.split("%")[0]) for x in m.group(1).split(",") if "%" in x]
                if loads:
                    cpu.append(statistics.mean(loads))
        if not ram:
            return None
        return {"ram_mb_max": max(ram), "gpu_pct_mean": round(statistics.mean(gpu), 1)
                if gpu else None, "cpu_pct_mean": round(statistics.mean(cpu), 1) if cpu else None}


def placement(url):
    try:
        ps = _get(url + "/api/ps")
        return [{"name": m.get("name"), "size_mb": round(m.get("size", 0) / 1e6),
                 "vram_mb": round(m.get("size_vram", 0) / 1e6)} for m in ps.get("models", [])]
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ report
def on_gpu(ph):
    """Did the model run on the GPU in this phase? From /api/ps (size_vram) when it
    was recorded; else the gpu/cpu phases are what they force; else unknown."""
    pl = ph.get("placement")
    if pl:
        return any((m.get("vram_mb") or 0) > 0 for m in pl)
    return {"gpu": True, "cpu": False}.get(ph["name"])


def placement_advice(res):
    """What this run says about where the model can run. Returns one line.

    A placement that breaks a limit is ruled out. One that keeps the limits and misses
    the targets is usable; which of two usable placements to run is a decision, and
    this only states the difference (2026-09-30: the GPU, by decision)."""
    phases = res["phases"]
    idle = next((p["stack"] for p in phases if p["name"] == "idle"), None)
    seen = {True: [], False: []}
    for ph in phases:
        where = on_gpu(ph) if ph["name"] != "idle" else None
        if where is None:
            continue
        threads = None
        if not where:
            threads = (res.get("cpu_threads") if ph["name"] == "cpu"
                       else (res.get("ask_options") or {}).get("num_thread"))
        seen[where].append(tiers_ok(ph["stack"], idle) + (threads,))

    def tally(rows):   # (limits kept, targets met) over every phase at one placement
        if not rows:
            return None, None
        return all(r[0] for r in rows), all(r[1] for r in rows)

    gl, gt = tally(seen[True])
    cl, ct = tally(seen[False])
    thr = next((r[2] for r in seen[False] if r[0] and r[2]), None)
    cpu_cfg = "llm_num_gpu: 0, llm_num_thread: {}".format(thr if thr else "<the threads measured>")
    cpu_said = ("the CPU was not measured here" if cl is None else
                "the CPU meets both ({})".format(cpu_cfg) if cl and ct else
                "the CPU keeps the limits and misses the targets ({})".format(cpu_cfg) if cl else
                "the CPU breaks a limit")
    if gl is True and gt:
        return "the GPU meets plan 3.6: llm_num_gpu: -1"
    if gl is True:
        return ("the GPU keeps the limits and misses plan 3.6's targets (usable; the targets "
                "are the optimization goal); {}".format(cpu_said))
    if gl is False and cl:
        return ("the GPU breaks a limit: move the model to the CPU ({})".format(cpu_cfg))
    if gl is False and cl is None:
        return "the GPU breaks a limit and the CPU was not measured here: add the cpu phase"
    if gl is False or (gl is None and cl is False):
        return ("no placement measured here keeps the limits: try --model qwen3:0.6b, "
                "then the model off the robot (llm_url, plan 3.6)")
    if cl:
        return "{}; the GPU was not measured here".format(cpu_said)
    return "no load phase with a known placement in this run"


def print_report(res):
    phases = res["phases"]
    print("\nstack log: {}".format(res.get("stack_log")))
    print("model: {} at {}".format(res.get("model"), res.get("url")))
    hdr = ("phase", "obs Hz min", "slow/deg", "lag p50/p95 ms", "infer p50/p99/max us",
           "LLM tok/s", "req", "RAM MB", "GPU %")
    print("\n" + " | ".join(hdr))
    for ph in phases:
        s, l, tg = ph["stack"], ph.get("llm") or {}, ph.get("tegra") or {}
        rate = (l["tokens"] / l["gen_s"]) if l.get("gen_s") else None
        print(" | ".join([
            ph["name"],
            "{:.1f}".format(s["obs_hz_min"]) if "obs_hz_min" in s else "-",
            "{}/{}".format(s.get("slow_windows"), s.get("degraded")),
            "{:.1f}/{:.1f}".format(s["lag_p50_median"], s["lag_p95_max"])
            if "lag_p95_max" in s else "-",
            "{:.0f}/{}/{}".format(s["infer_p50_median"], s["infer_p99_max"], s["infer_max_max"])
            if "infer_p99_max" in s else "-",
            "{:.1f}".format(rate) if rate else "-",
            str(l.get("requests", "-")),
            str(tg.get("ram_mb_max", "-")), str(tg.get("gpu_pct_mean", "-"))]))
    idle = next((p["stack"] for p in phases if p["name"] == "idle"), None)
    print()
    all_ok, missed = True, []
    for ph in phases:
        if ph["name"] == "idle":
            continue
        for tier, name, ok, detail in verdict(ph["stack"], idle):
            if tier == LIMIT:
                all_ok = all_ok and ok
                word = "PASS" if ok else "FAIL"
            else:
                word = "met" if ok else "MISS"
                if not ok and ph["name"] not in missed:
                    missed.append(ph["name"])
            print("  {:<6} {:<6} {:<6} {:<44} {}".format(word, tier, ph["name"], name, detail))
    print("\nlimits: {}".format(
        "every load phase keeps them" if all_ok else
        "BROKEN -- see the FAIL lines; that placement cannot be used"))
    print("targets (plan 3.6): {}".format(
        "met in every load phase" if not missed else "missed in " + ", ".join(missed)))
    if res.get("ask_options") is not None:
        print("ask phase placement: {}".format(
            json.dumps(res["ask_options"], sort_keys=True) if res["ask_options"]
            else "Ollama's default (the GPU)"))
    print("placement: {}".format(placement_advice(res)))
    return all_ok


# ----------------------------------------------------------------- selftest
def synth(spec, placements=None, **extra):
    """A synthetic run: stack log lines for each (phase, obs Hz, lag p50, lag p95,
    infer p50, infer p99), summarized exactly as a real run is."""
    t0 = 1790000000.0
    lines, phases = [], []
    t = t0
    for name, hz, l50, l95, i50, i99 in spec:
        start, end = t, t + 60.0
        phases.append({"name": name, "start": start, "end": end})
        for k in range(1, 13):
            ts = start + 5.0 * k
            if ts > end:
                break
            lines.append("[r1_hw_bridge_node-1] [INFO] [{:.9f}] [r1_hw_bridge]: RUNNING | obs "
                         "{:.1f} Hz (want 50) | cmd 500.0 Hz (want 500) | output=off | "
                         "kp_scale=1.30 | crc_fail=0 | setpoint lag p50={:.1f} p95={:.1f} "
                         "max={:.1f} ms (0.03 steps; trained 0-20)".format(
                             ts, hz, l50, l95, l95 + 1.0))
            lines.append("[r1_policy_node-2] [INFO] [{:.9f}] [r1_policy_node]: 250 cycles in "
                         "5.0s (50.0 Hz) | infer us p50={} p95={} p99={} max={} | budget "
                         "20000 us | failures=0".format(ts + 0.2, i50, i99 - 50, i99, i99 + 100))
        if hz < 45:
            lines.append("[r1_hw_bridge_node-1] [WARN] [{:.9f}] [r1_hw_bridge]: observation "
                         "rate 43.0 Hz below 45.0 Hz (1/3)".format(start + 20))
        t = end + 5.0
    ev = parse_log(lines)
    res = dict({"stack_log": "(synthetic)", "model": "none", "url": "-", "phases": []}, **extra)
    for i, ph in enumerate(phases):
        res["phases"].append({"name": ph["name"], "start": ph["start"], "end": ph["end"],
                              "placement": (placements or {}).get(i),
                              "stack": summarize_phase(ev, ph["start"], ph["end"])})
    return res


def selftest():
    fails = 0
    res = synth([("idle", 50.0, 0.6, 1.1, 455, 700), ("gpu", 50.0, 0.7, 1.4, 480, 1500),
                 ("cpu", 43.0, 0.9, 25.0, 600, 3100), ("idle", 50.0, 0.6, 1.1, 455, 700)],
                cpu_threads=4)
    for ph in res["phases"]:
        s = ph["stack"]
        if s["bridge_samples"] < 9 or s["policy_samples"] < 11:
            print("FAIL: too few samples in {}: {}".format(ph["name"], s))
            fails += 1
    idle = res["phases"][0]["stack"]
    if tiers_ok(res["phases"][1]["stack"], idle) != (True, True):
        print("FAIL: the gpu phase should meet both tiers: {}".format(
            verdict(res["phases"][1]["stack"], idle)))
        fails += 1
    c = dict((n, ok) for _, n, ok, _ in verdict(res["phases"][2]["stack"], idle))
    broken = sorted(n.split(",")[0] for n, ok in c.items() if not ok)
    if broken != ["infer p99 <= 2000 us (plan 3.6)", "lag p95 < one step (20 ms)",
                  "lag p95 <= idle + 1 ms (plan 3.6)", "obs >= 45 Hz"]:
        print("FAIL: the cpu phase breaks obs and lag and misses both targets: {}".format(c))
        fails += 1
    if res["phases"][2]["stack"]["slow_windows"] != 1:
        print("FAIL: the slow window was not counted")
        fails += 1
    if print_report(res):
        print("FAIL: a broken limit must give a failing exit status")
        fails += 1
    if "the GPU meets plan 3.6" not in placement_advice(res):
        print("FAIL: GPU meets both -> keep it: {}".format(placement_advice(res)))
        fails += 1

    # The robot on 2026-09-30: GPU decode holds the policy's TensorRT inference at
    # ~5 ms (p99 > 2 ms) and the lag at ~5 ms -- inside the limits, outside the targets;
    # 4 CPU threads leave both near idle.
    gpu = {"name": "qwen3:1.7b", "size_mb": 1466, "vram_mb": 1466}
    cpu = {"name": "qwen3:1.7b", "size_mb": 1645, "vram_mb": 0}
    real = [("idle", 50.0, 0.6, 0.6, 455, 486), ("gpu", 50.0, 3.7, 5.1, 3519, 5010),
            ("ask", 50.0, 3.9, 5.2, 3635, 5014), ("cpu", 50.0, 0.8, 0.9, 495, 542),
            ("idle", 50.0, 0.6, 0.6, 454, 489)]
    r2 = synth(real, {1: [gpu], 2: [gpu], 3: [cpu]}, cpu_threads=4, ask_options={})
    a2 = placement_advice(r2)
    if not ("keeps the limits and misses plan 3.6's targets" in a2
            and "the CPU meets both (llm_num_gpu: 0, llm_num_thread: 4)" in a2):
        print("FAIL: GPU usable but off target, CPU meets both: {}".format(a2))
        fails += 1
    if not print_report(r2):
        print("FAIL: kept limits must give a passing exit status even with missed targets")
        fails += 1
    r3 = synth(real[:3] + real[4:], {1: [gpu], 2: [gpu]}, cpu_threads=4, ask_options={})
    if "the CPU was not measured here" not in placement_advice(r3):
        print("FAIL: no CPU phase -> say so: {}".format(placement_advice(r3)))
        fails += 1
    # An ask phase that ran on the CPU (mission.yaml moved it there) counts as the CPU.
    r4 = synth([real[0], real[1], ("ask", 50.0, 0.8, 1.0, 497, 560), real[4]],
               {1: [gpu], 2: [cpu]}, cpu_threads=4, ask_options={"num_gpu": 0, "num_thread": 4})
    if "the CPU meets both (llm_num_gpu: 0, llm_num_thread: 4)" not in placement_advice(r4):
        print("FAIL: an ask phase on the CPU counts as the CPU: {}".format(
            placement_advice(r4)))
        fails += 1
    # A GPU that breaks a limit is ruled out when the CPU keeps them.
    r5 = synth([real[0], ("gpu", 50.0, 12.0, 25.0, 3500, 5000), real[3], real[4]],
               {1: [gpu], 2: [cpu]}, cpu_threads=4)
    if "move the model to the CPU (llm_num_gpu: 0, llm_num_thread: 4)" not in placement_advice(r5):
        print("FAIL: GPU breaks a limit -> the CPU: {}".format(placement_advice(r5)))
        fails += 1
    print("\nselftest: {}".format("OK" if not fails else "{} FAILURES".format(fails)))
    return 1 if fails else 0


# --------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stack-log", help="the tee'd output of `ros2 launch ... r1_stack`")
    ap.add_argument("--url", default=os.environ.get("R1_LLM_URL", "http://127.0.0.1:11434"))
    ap.add_argument("--model", default=os.environ.get("R1_LLM_MODEL", "qwen3:1.7b"))
    ap.add_argument("--phases", default="idle,gpu,ask,cpu,idle",
                    help="comma list of idle | gpu | ask | cpu")
    ap.add_argument("--phase-s", type=float, default=60.0)
    ap.add_argument("--cpu-threads", type=int, default=4,
                    help="threads for the cpu phase (0 = Ollama's default: every core)")
    ap.add_argument("--ask-options", default=None,
                    help="placement for the ask phase as Ollama options, e.g. "
                         "'{\"num_gpu\": 0, \"num_thread\": 4}' (default: mission.yaml's, "
                         "which is what `ask` uses)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-tegrastats", action="store_true")
    ap.add_argument("--report", default=None, help="re-print a saved result")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.report:
        return 0 if print_report(json.loads(pathlib.Path(args.report).read_text())) else 1
    if not args.stack_log:
        ap.error("--stack-log is required")
    log = pathlib.Path(os.path.expanduser(args.stack_log))
    if not log.is_file():
        ap.error("{} does not exist -- start the stack with its output tee'd there".format(log))
    recent = parse_log(log.read_text(errors="replace").splitlines()[-400:])
    if not recent["bridge"] or not recent["policy"]:
        print("WARNING: no bridge/policy timing lines in the last 400 lines of {}.\n"
              "  Is the stack running, and was it started with PYTHONUNBUFFERED=1 and\n"
              "  `2>&1 | tee` into this file? Continuing; the summary may be empty.".format(log))
    try:
        ver = _get(args.url + "/api/version").get("version")
    except Exception as exc:  # noqa: BLE001
        print("the model server does not answer at {}: {}".format(args.url, exc))
        return 3
    out = pathlib.Path(os.path.expanduser(args.out or "~/orin_commissioning/coexist_{}.json".format(
        time.strftime("%H%M%S"))))
    out.parent.mkdir(parents=True, exist_ok=True)
    names = [x.strip() for x in args.phases.split(",") if x.strip()]
    ask_opts = (json.loads(args.ask_options) if args.ask_options
                else configured_ask_options())
    print("server {} at {}, model {}; phases {} x {:.0f} s; ask placement {}".format(
        ver, args.url, args.model, names, args.phase_s,
        json.dumps(ask_opts, sort_keys=True) if ask_opts else "Ollama's default (the GPU)"))

    tegra = None if args.no_tegrastats else Tegrastats()
    phases = []
    for name in names:
        opts = phase_options(name, args.cpu_threads)
        if name in ("gpu", "cpu"):
            print("[{}] warm-up (loads the model {})...".format(
                name, "on the GPU" if name == "gpu" else "on the CPU"))
            try:
                _post(args.url + "/api/generate",
                      {"model": args.model, "prompt": "ok", "stream": False,
                       "keep_alive": "30m", "options": dict(opts, num_predict=1)}, 300)
            except Exception as exc:  # noqa: BLE001
                print("  warm-up failed: {}".format(exc))
        elif name == "ask":
            print("[ask] warm-up (one real request: the prompt, the schema, the placement)...")
            try:
                warm_ask(args.url, args.model, ask_opts)
            except Exception as exc:  # noqa: BLE001
                print("  warm-up failed: {}".format(exc))
        start = time.time()
        until = start + args.phase_s
        stats = {"requests": 0, "tokens": 0, "gen_s": 0.0, "errors": 0}
        print("[{}] {:.0f} s from {}".format(name, args.phase_s, time.strftime("%H:%M:%S")))
        if name in ("gpu", "cpu"):
            run_generate_load(args.url, args.model, opts, until, stats)
        elif name == "ask":
            run_ask_load(args.url, args.model, until, stats, ask_opts)
        else:
            time.sleep(args.phase_s)
        end = time.time()
        phases.append({"name": name, "start": start, "end": end,
                       "llm": stats if name != "idle" else None,
                       "placement": placement(args.url) if name != "idle" else None})
        if name != "idle":
            rate = stats["tokens"] / stats["gen_s"] if stats["gen_s"] else 0.0
            print("  {} requests, {:.1f} tok/s, {} errors; placement {}".format(
                stats["requests"], rate, stats["errors"], phases[-1]["placement"]))

    print("waiting 8 s for the last timing lines to reach the log...")
    time.sleep(8.0)
    if tegra:
        tegra.stop()
    ev = parse_log(log.read_text(errors="replace").splitlines())
    for ph in phases:
        ph["stack"] = summarize_phase(ev, ph["start"], ph["end"])
        ph["tegra"] = tegra.summarize(ph["start"], ph["end"]) if tegra else None
    res = {"stack_log": str(log), "url": args.url, "model": args.model, "server_version": ver,
           "cpu_threads": args.cpu_threads, "ask_options": ask_opts, "phase_s": args.phase_s,
           "phases": phases, "when": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    out.write_text(json.dumps(res, indent=1))
    ok = print_report(res)
    print("saved {}".format(out))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
