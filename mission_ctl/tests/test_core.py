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
from r1_mission import node as N        # noqa: E402  (rclpy is imported lazily)
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
check("capability prose (English, the LLM's input) names the pinned axis",
      "cannot move sideways" in TRAINED.describe(), TRAINED.describe())
check("capability prose says it cannot walk backward",
      "cannot walk backward" in TRAINED.describe())
check("operator view still available in Chinese", "不能横移" in TRAINED.describe("zh"))
wide = P.Envelope([0.0, 1.0], [-0.3, 0.3], [-0.5, 0.5])
check("a retrained policy with vy changes the prose with no hand edit",
      "can move sideways" in wide.describe() and "cannot move sideways" not in wide.describe())

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
                               P.Limits(v_cal=None)), "no speed is configured")
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
check("report carries the turn's expected time next to its actual time",
      rep["executed"][1].get("expected_s", 0) > rep["executed"][1].get("commanded_s", 0),
      rep["executed"][1])

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

# The live node starts with no status at all. That must mean "wait", not "abort".
e9 = ex.Executor(prims, lim)
o = e9.step(0.0, yaw=0.0, bridge_state="UNKNOWN")
o = e9.step(0.1, yaw=0.0, bridge_state="UNKNOWN")
check("no status yet = wait, not abort", e9.state == ex.IDLE, (e9.state, e9.abort_reason))
o = e9.step(0.9, yaw=0.0, bridge_state="RUNNING")
check("starts once RUNNING arrives", e9.state == ex.RUNNING, e9.state)
e10 = ex.Executor(prims, lim)
for k in range(0, 120):
    o = e10.step(k * 0.1, yaw=0.0, bridge_state="UNKNOWN")
    if e10.state == ex.ABORTED:
        break
check("never RUNNING -> bounded wait, then abort naming the ROS env",
      e10.state == ex.ABORTED and "no ~/status" in (e10.abort_reason or ""),
      e10.abort_reason)
check("start timeout is ~10 s", 9.9 <= k * 0.1 <= 10.2, k * 0.1)
e11, _, _, _ = simulate(prims, lim, degrade_at=None)
e12 = ex.Executor(prims, lim)
e12.step(0.0, yaw=0.0, bridge_state="RUNNING"); e12.step(0.1, yaw=0.0, bridge_state="RUNNING")
e12.step(0.2, yaw=0.0, bridge_state="NO_STATUS")
check("status going quiet mid-run aborts", e12.state == ex.ABORTED
      and "NO_STATUS" in e12.abort_reason, e12.abort_reason)
e13 = ex.Executor(prims, lim)
e13.step(0.0, yaw=0.0, bridge_state="DEGRADED")
check("refuses to start on a DEGRADED bridge", e13.state == ex.ABORTED, e13.state)

e6 = ex.Executor(prims, lim)
out = e6.step(0.0, yaw=0.0, bridge_state="WAITING_POLICY")
check("will not start before the bridge is RUNNING",
      out.state == ex.IDLE and out.vx == 0.0, out.state)

print("\n== assumed ground speed (v_cal: commanded, decided 2026-09-28) ==")
lim_a = P.Limits(v_cal_assumed=True, cruise_vx=0.4)
pa, _, ma = P.compile_plan(P.parse_script("walk 10m@0.5; walk 4s@0.3"), TRAINED, lim_a)
check("distance -> time at the commanded speed", abs(pa[0].duration_s - 20.0) < 1e-9,
      pa[0].duration_s)
check("no error bar is invented for an unmeasured speed",
      pa[0].distance_err_m is None and pa[1].distance_err_m is None)
check("a timed walk still estimates its distance", abs(pa[1].distance_m - 1.2) < 1e-9,
      pa[1].distance_m)
check("distance budget still counts assumed metres", abs(ma - 11.2) < 1e-9, ma)
check("the plan line says the speed is assumed", "not measured" in pa[0].summary(),
      pa[0].summary())
ea, _, _, _ = simulate(pa, lim_a)
ra = ea.report()["executed"][0]
check("report labels the distance as assumed and gives no est_err_m",
      "assumed" in ra.get("distance_source", "") and "est_err_m" not in ra, ra)
pb = P.compile_plan(P.parse_script("walk 10m@0.5"), TRAINED,
                    P.Limits(v_cal=0.2, v_cal_assumed=True))[0]
check("assumed wins over a stray v_cal number", abs(pb[0].duration_s - 20.0) < 1e-9)
check("config 'v_cal: commanded' -> assumed", N.limits_from_cfg({"v_cal": "commanded"})
      .v_cal_assumed)
lu = N.limits_from_cfg({})
check("config without v_cal -> unmeasured, not assumed",
      lu.v_cal is None and not lu.v_cal_assumed)
lm = N.limits_from_cfg({"v_cal": "0.35"})
check("config with a number -> measured", lm.v_cal == 0.35 and not lm.v_cal_assumed)
refuses("unmeasured still refuses distance",
        lambda: P.compile_plan(P.parse_script("walk 3m"), TRAINED, lu), "no speed")
cap = N.capability_text(TRAINED, lim_a)
check("capability tells the LLM distance is approximate and unmeasured",
      "given distance" in cap and "not been measured" in cap, cap)

print("\n== turn timing: the taper is part of the estimate ==")
# 2026-09-28 sweep: the timeout was 2x the bare angle/rate, which is only ~1.7x
# of how long the executor really takes, and four of ten gantry points hit it.
for deg, w in ((20, 0.15), (90, 0.4), (180, 0.4), (360, 0.4)):
    pt = P.compile_plan(P.parse_script("turn left {}@{}".format(deg, w)), TRAINED, lim)[0]
    et, _, _, _ = simulate(pt, lim)
    act = et.report()["executed"][0]["actual_s"]
    check("expected_turn_s({}deg @{}) = {:.2f}s matches the executor ({:.2f}s)"
          .format(deg, w, pt[0].expected_s, act), abs(pt[0].expected_s - act) < 0.1)
pt, tt, _ = P.compile_plan(P.parse_script("turn left 180@0.4"), TRAINED, lim)
check("plan total uses the executed turn time, not the bare rate",
      abs(tt - pt[0].expected_s) < 1e-9 and tt > pt[0].duration_s + 1.0, (tt, pt[0]))
eslow, _, _, _ = simulate(pt, lim, turn_gain=0.55)
check("a robot turning at 55% of the command still finishes (old timeout aborted it)",
      eslow.state == ex.DONE, eslow.abort_reason)
estuck, _, tstuck, _ = simulate(pt, lim, turn_gain=0.0)
check("a robot that will not turn still times out, at 2x the executed estimate",
      estuck.state == ex.ABORTED and abs(tstuck - 2 * pt[0].expected_s) < 0.2,
      (estuck.state, tstuck))

print("\n" + "-" * 50)
print("pass {}  fail {}".format(len(PASS), len(FAIL)))
sys.exit(1 if FAIL else 0)
