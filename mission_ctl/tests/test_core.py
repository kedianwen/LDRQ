#!/usr/bin/env python3
"""Tests for the mission core. No ROS, no pytest -- `python3 tests/test_core.py`.

pytest is not assumed because the robot's apt is broken and its Python is 3.8;
a test suite that cannot run on the machine that matters is not a test suite.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r1_mission import executor as ex   # noqa: E402
from r1_mission import plan as P        # noqa: E402
from r1_mission import yaw as Y         # noqa: E402

TRAINED = P.Envelope([0.0, 1.0], [0.0, 0.0], [-0.5, 0.5], source="as trained")
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print("  {}  {}{}".format("ok  " if cond else "FAIL", name,
                              "" if cond else "  <- " + str(detail)))


def refuses(name, fn, want):
    try:
        fn()
    except P.PlanError as e:
        check(name, want.lower() in str(e).lower(), "refused, but: " + str(e).split("\n")[0])
        return
    check(name, False, "was ACCEPTED")


def simulate(prims, limits, hz=50.0, turn_gain=1.0, degrade_at=None, max_s=300.0):
    """Run the executor against a robot that turns at exactly the commanded wz
    (scaled by turn_gain, so 0.0 models a robot that does not turn at all)."""
    e = ex.Executor(prims, limits)
    dt, t, acc_yaw = 1.0 / hz, 0.0, 0.0
    trace = []
    while t < max_s:
        state = "RUNNING"
        if degrade_at is not None and t >= degrade_at:
            state = "DEGRADED"
        out = e.step(t, yaw=acc_yaw, bridge_state=state)
        trace.append((t, out.vx, out.wz, out.state))
        if out.state in (ex.DONE, ex.ABORTED):
            break
        acc_yaw += out.wz * turn_gain * dt
        t += dt
    return e, trace, t, acc_yaw


print("\n== script parsing ==")
steps = P.parse_script("walk 5s@0.4; turn left 90; walk 10m@0.4; stand 3s")
check("four steps parsed", len(steps) == 4, steps)
check("walk seconds", steps[0]["duration_s"] == 5.0 and steps[0]["vx"] == 0.4)
check("turn degrees + direction", steps[1]["yaw_deg"] == 90.0
      and steps[1]["direction"] == "left")
check("walk metres", steps[2]["distance_m"] == 10.0)
check("turn by seconds is not read as degrees",
      P.parse_script("turn right 4s")[0].get("duration_s") == 4.0)
check("comments and blank lines skipped",
      len(P.parse_script("# hi\n\nwalk 3s\n")) == 1)
refuses("gibberish is refused", lambda: P.parse_script("fly 3s"), "cannot parse")
refuses("strafe has no syntax at all", lambda: P.parse_script("strafe left 2m"),
        "cannot parse")

print("\n== envelope ==")
lim = P.Limits(v_cal=0.35, cruise_vx=0.4, cruise_wz=0.4)
prims, tot_s, tot_m = P.compile_plan(
    P.parse_script("walk 5s@0.4; turn left 90; walk 10m@0.4"), TRAINED, lim)
check("three primitives compiled", len(prims) == 3, prims)
check("distance -> duration via v_cal",
      abs(prims[2].duration_s - 10.0 / 0.35) < 1e-6, prims[2].duration_s)
check("distance carries an error bar", prims[2].distance_err_m is not None
      and prims[2].distance_err_m > 0)
check("duration-only walk still reports an estimated distance",
      prims[0].distance_m is not None and abs(prims[0].distance_m - 0.35 * 5) < 1e-9)
check("closed-loop turn is flagged", prims[1].closed_loop)
check("capability prose names the pinned axis",
      "不能横移" in TRAINED.describe(), TRAINED.describe())

refuses("vx above the trained range",
        lambda: P.compile_plan(P.parse_script("walk 5s@1.5"), TRAINED, lim), "outside")
refuses("reverse walking",
        lambda: P.compile_plan([{"op": "walk", "vx": -0.3, "duration_s": 5.0}],
                               TRAINED, lim), "outside")
refuses("wz above the trained range",
        lambda: P.compile_plan(P.parse_script("turn left 90@0.9"), TRAINED, lim),
        "outside")
refuses("pinned axis is refused, not clamped",
        lambda: TRAINED.check("vy", 0.2, "strafe"), "pinned to 0")
refuses("primitive under the 2 s floor",
        lambda: P.compile_plan(P.parse_script("walk 1s@0.4"), TRAINED, lim), "floor")
refuses("a 90 deg turn too fast to last 2 s",
        lambda: P.compile_plan(P.parse_script("turn left 30@0.5"), TRAINED, lim),
        "floor")
refuses("distance with no calibrated speed",
        lambda: P.compile_plan(P.parse_script("walk 10m@0.4"), TRAINED,
                               P.Limits(v_cal=None)), "no calibrated speed")
refuses("total duration budget",
        lambda: P.compile_plan(P.parse_script("walk 100s@0.4; walk 100s@0.4"),
                               TRAINED, lim), "budget")
refuses("total distance budget",
        lambda: P.compile_plan(P.parse_script("walk 25m@0.4; walk 25m@0.4"),
                               TRAINED, lim), "budget")
refuses("too many primitives",
        lambda: P.compile_plan(P.parse_script("; ".join(["walk 3s"] * 25)),
                               TRAINED, lim), "budget")

print("\n== json front end (the LLM's entry point) ==")
good = {"plan": [{"op": "walk", "vx": 0.4, "duration_s": 5.0},
                 {"op": "turn", "direction": "left", "yaw_deg": 90, "wz": 0.4},
                 {"op": "walk", "vx": 0.4, "distance_m": 10.0},
                 {"op": "stop"}]}
jp, js, jm = P.plan_from_json(good, TRAINED, lim)
check("the worked example compiles", len(jp) == 4, jp)
check("same result as the script form",
      abs(jp[2].duration_s - prims[2].duration_s) < 1e-9)
refuses("invented op", lambda: P.plan_from_json(
    {"plan": [{"op": "backflip"}]}, TRAINED, lim), "op must be one of")
refuses("walk with both duration and distance", lambda: P.plan_from_json(
    {"plan": [{"op": "walk", "duration_s": 3, "distance_m": 3}]}, TRAINED, lim),
    "exactly one")
refuses("walk with neither", lambda: P.plan_from_json(
    {"plan": [{"op": "walk"}]}, TRAINED, lim), "exactly one")
refuses("smuggled field", lambda: P.plan_from_json(
    {"plan": [{"op": "walk", "duration_s": 3, "kp_scale": 0.5}]}, TRAINED, lim),
    "unknown field")
refuses("vy smuggled in as a field", lambda: P.plan_from_json(
    {"plan": [{"op": "walk", "duration_s": 3, "vy": 0.3}]}, TRAINED, lim),
    "unknown field")
refuses("vx out of range from json", lambda: P.plan_from_json(
    {"plan": [{"op": "walk", "vx": 5.0, "duration_s": 3}]}, TRAINED, lim), "outside")
refuses("turn with no direction", lambda: P.plan_from_json(
    {"plan": [{"op": "turn", "yaw_deg": 90}]}, TRAINED, lim), "direction")
refuses("not an object", lambda: P.plan_from_json("[1,2,3]", TRAINED, lim),
        "expected an object")
refuses("malformed json text", lambda: P.plan_from_json("{oops", TRAINED, lim),
        "not valid json")

print("\n== yaw ==")
def feed(tracker, degrees):
    for deg in degrees:
        a = math.radians(((deg + 180.0) % 360.0) - 180.0)
        tracker.update((math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)))
    return math.degrees(tracker.total)

# 170 -> 179 -> -179 (i.e. 181) -> -170 (i.e. 190) is a +20 deg turn that happens
# to cross the wrap. Without unwrapping it reads as -340.
got = feed(Y.YawTracker(), [170, 179, -179, -170])
check("crossing +-pi accumulates +20, not -340", abs(got - 20.0) < 1e-6, got)
# The first sample is the reference, so 0,100,...,400 accumulates 400.
got = feed(Y.YawTracker(), [0, 100, 200, 300, 400])
check("accumulates past 360 deg", abs(got - 400.0) < 1e-6, got)
got = feed(Y.YawTracker(), [0, -100, -200, -300])
check("accumulates negative past -180", abs(got + 300.0) < 1e-6, got)

print("\n== executor ==")
e, trace, t_end, yaw_end = simulate(prims, lim)
check("plan completes", e.state == ex.DONE, e.state)
check("ends at zero velocity", trace[-1][1] == 0.0 and trace[-1][2] == 0.0)
turn_rec = e.records[1]
check("closed-loop turn lands on the target, not on the tolerance band",
      abs(turn_rec.achieved_deg - 90.0) <= 0.5, turn_rec.achieved_deg)
check("walk 1 lasted its commanded time",
      abs(e.records[0].actual_s - 5.0) < 0.05, e.records[0].actual_s)
check("walk 2 lasted its commanded time",
      abs(e.records[2].actual_s - 10.0 / 0.35) < 0.05, e.records[2].actual_s)
check("ramp: first commanded vx is not the full cruise speed",
      0.0 < [x for x in trace if x[1] > 0][0][1] < 0.4)
check("ramp reaches cruise speed", max(x[1] for x in trace) >= 0.4 - 1e-6)
rep = e.report()
check("report has one entry per primitive", len(rep["executed"]) == len(prims))
check("report marks the distance as an estimate",
      "distance_source" in rep["executed"][2], rep["executed"][2])
check("report gives the turn a measured source",
      rep["executed"][1].get("source") == "imu_yaw")

e2, tr2, _, _ = simulate(prims, lim, degrade_at=3.0)
check("bridge DEGRADED aborts the plan", e2.state == ex.ABORTED, e2.state)
check("abort zeroes the command", tr2[-1][1] == 0.0 and tr2[-1][2] == 0.0)
check("abort reason names the bridge", "bridge" in (e2.abort_reason or ""),
      e2.abort_reason)

e3, _, _, _ = simulate(prims, lim, turn_gain=0.0)
check("a robot that will not turn times out rather than spinning forever",
      e3.state == ex.ABORTED and "timed out" in (e3.abort_reason or ""),
      e3.abort_reason)

e4, tr4, _, _ = simulate(
    P.compile_plan(P.parse_script("turn right 90"), TRAINED, lim)[0], lim)
check("right turn goes negative", min(x[2] for x in tr4) < 0
      and max(x[2] for x in tr4) <= 0.0 + 1e-9)
check("right turn lands on the target",
      abs(e4.records[0].achieved_deg + 90.0) <= 0.5, e4.records[0].achieved_deg)

e5, tr5, _, _ = simulate(P.compile_plan(P.parse_script("stand 3s"), TRAINED, lim)[0], lim)
check("stand commands exactly zero throughout",
      all(abs(x[1]) < 1e-12 and abs(x[2]) < 1e-12 for x in tr5))

e7, tr7, _, _ = simulate(jp, lim)     # the worked example, ending in "stop"
check("a plan whose last step is 'stop' completes", e7.state == ex.DONE, e7.state)
check("the stop record is present", e7.report()["executed"][-1]["op"] == "stop")
e8, _, _, _ = simulate(
    P.compile_plan(P.parse_script("walk 3s; stop; walk 3s"), TRAINED, lim)[0], lim)
check("a mid-plan 'stop' does not end the plan early", e8.state == ex.DONE
      and len(e8.report()["executed"]) == 3, e8.report())

e6 = ex.Executor(prims, lim)
out = e6.step(0.0, yaw=0.0, bridge_state="WAITING_POLICY")
check("will not start before the bridge is RUNNING",
      out.state == ex.IDLE and out.vx == 0.0, out.state)

print("\n" + "-" * 50)
print("pass {}  fail {}".format(len(PASS), len(FAIL)))
sys.exit(1 if FAIL else 0)
