#!/usr/bin/env python3
"""The real-robot half of the kp_scale stability-domain sweep (stage A).

INT8 was cut on 2026-09-26, so FR-R2's controlled perturbation is the actuator
gain error instead of numeric precision. The methodology is unchanged: sweep one
knob across simulation, spot-check the same knob on the robot, and put both on one
axis. This tool is the robot side -- it does not launch anything and it does not
move the robot. It plans the points, then reduces the per-point recordings that
walk_metrics.py saved into the one table and the one number stage A is asked for.

  # before going near the robot: the checklist, in the order the points are run
  python3 gain_sweep_real.py --plan

  # after the session, from the saved recordings
  python3 gain_sweep_real.py --collect ~/orin_commissioning/gain_sweep
  python3 gain_sweep_real.py --collect ~/orin_commissioning/gain_sweep \
      --sim ~/R1process/outputs/gain_sweep/<date>/results.json

Each recording must carry its gain in the note as `kp=<value>`; a recording that
does not is REFUSED rather than guessed at. The whole point of the sweep is that
one number, and a mislabelled point is worse than a missing one.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import walk_metrics as W   # noqa: E402  -- helpers, and the gains it already parses

# Run 1.0 first (it is the trained controller and the baseline every other point
# is judged against), then step outward alternately so that if the session is cut
# short the surviving points still bracket the baseline.
PLAN_ORDER = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6, 0.5]

TRACKING_FACTOR = 1.5   # the stability-domain definition, fixed before measuring
TILT_LIMIT_DEG = 20.0   # beyond this the robot is not walking, it is falling


def parse_kp(note):
    for token in str(note).replace(",", " ").split():
        if token.startswith("kp="):
            try:
                return float(token[3:])
            except ValueError:
                return None
    return None


def tilt_deg(grav):
    """Angle between the measured gravity direction and the torso-upright one.

    Against UPRIGHT_GRAVITY, not (0,0,-1): the IMU is not mounted level and the
    ideal vector reports ~3.5 deg on a robot standing perfectly straight.
    """
    u = W.UPRIGHT_GRAVITY
    nu = math.sqrt(sum(c * c for c in u))
    ng = math.sqrt(sum(c * c for c in grav))
    if nu == 0 or ng == 0:
        return float("nan")
    dot = sum(a * b for a, b in zip(u, grav)) / (nu * ng)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def reduce_point(blob, defaults, kp_gains, names, groups):
    """Only the numbers the stability-domain definition needs. walk_metrics.py
    remains the tool for the full five-section read of a single point."""
    t0, t1 = blob["t_start"], blob["t_end"]
    out = {"window_s": t1 - t0}

    # survival: longest continuous RUNNING, and whether the window ended while
    # still RUNNING (which makes it a LOWER bound, not a measurement)
    best, cur, ended_running, degraded = 0.0, None, False, None
    for t, text in blob.get("status", []):
        if str(text).startswith("RUNNING"):
            if cur is None:
                cur = t
        else:
            if cur is not None:
                best = max(best, t - cur)
                cur = None
            if degraded is None and str(text).startswith("DEGRADED"):
                degraded = str(text)
    if cur is not None:
        best = max(best, t1 - cur)
        ended_running = True
    out["running_s"] = best
    out["ended_running"] = ended_running
    out["degraded"] = degraded
    out["have_status"] = bool(blob.get("status"))

    # tilt
    tilts = [tilt_deg(v[7:10]) for _, v in blob.get("imu", []) if len(v) >= 10]
    out["tilt_max_deg"] = max(tilts) if tilts else float("nan")

    # leg tracking error, rms. cmd_debug is absolute commanded joint angle,
    # joint_pos is relative to the default pose -- hence the + defaults[j].
    cmd, jpos = blob.get("cmd", []), blob.get("jpos", [])
    leg = [j for j, g in enumerate(groups) if g in ("legs", "ankles")]
    sq, n = 0.0, 0
    if cmd and jpos and leg:
        jt = [t for t, _ in jpos]
        jv = [v for _, v in jpos]
        for t, c in cmd:
            m = W.nearest(jt, jv, t)
            if m is None:
                continue
            for j in leg:
                if j < len(c) and j < len(m):
                    e = c[j] - (m[j] + defaults[j])
                    sq += e * e
                    n += 1
    out["leg_track_rms"] = math.sqrt(sq / n) if n else float("nan")

    # torque headroom: worst |tau| as a fraction of that joint's rating
    tau_limit, _, _ = W.parse_hpp_array("kTauLimit")
    worst, worst_j = 0.0, None
    for _, v in blob.get("tau", []):
        for j, x in enumerate(v):
            if j >= len(tau_limit) or tau_limit[j] <= 0:
                continue
            frac = abs(x) / tau_limit[j]
            if frac > worst:
                worst, worst_j = frac, j
    out["tau_frac_max"] = worst
    out["tau_worst_joint"] = names[worst_j] if worst_j is not None else None
    out["tau_all_zero"] = all(all(x == 0.0 for x in v) for _, v in blob.get("tau", [])) \
        and bool(blob.get("tau"))

    # action chatter, as the implied dominant frequency (flips / 2)
    acts = [v for _, v in blob.get("action", [])]
    flips, steps, dur = 0, 0, max(1e-9, t1 - t0)
    for i in range(2, len(acts)):
        for j in range(min(len(acts[i]), len(acts[i - 1]), len(acts[i - 2]))):
            d1 = acts[i - 1][j] - acts[i - 2][j]
            d2 = acts[i][j] - acts[i - 1][j]
            if d1 * d2 < 0:
                flips += 1
            steps += 1
    out["chatter_hz"] = (flips / max(1, steps)) * (len(acts) / dur) / 2.0 if steps else 0.0
    return out


def collect(directory, sim_path=None):
    files = sorted(pathlib.Path(directory).glob("*.json"))
    if not files:
        raise SystemExit("[fail] no *.json recordings in " + str(directory))

    defaults, _terms, _an = W.load_spec()
    kp_gains, names, groups = W.parse_hpp_array("kKp")

    points, refused = {}, []
    for f in files:
        try:
            blob = json.loads(f.read_text())
        except ValueError as exc:
            refused.append((f.name, "not valid JSON ({})".format(exc)))
            continue
        if "t_start" not in blob or "status" not in blob:
            refused.append((f.name, "not a walk_metrics recording"))
            continue
        kp = parse_kp(blob.get("note", ""))
        if kp is None:
            refused.append((f.name, "no 'kp=<value>' in its note -- refusing to guess "
                                    "which gain this point was recorded at"))
            continue
        points.setdefault(kp, []).append((f.name, reduce_point(
            blob, defaults, kp_gains, names, groups)))

    for name, why in refused:
        print("REFUSED  {}: {}".format(name, why))
    if refused:
        print()
    if not points:
        raise SystemExit("[fail] no usable points")

    print("=== per point ({} recordings at {} gain(s)) ===".format(
        sum(len(v) for v in points.values()), len(points)))
    print("{:>6}  {:>8}  {:>9}  {:>9}  {:>9}  {:>9}  {}".format(
        "kp", "run s", "tilt deg", "leg rms", "tau %rat", "chat Hz", "verdict"))

    baseline = None
    rows = []
    for kp in sorted(points, reverse=True):
        for fname, m in points[kp]:
            rows.append((kp, fname, m))
    # the kp=1.0 point is the baseline every other point is compared against
    for kp, _f, m in rows:
        if abs(kp - 1.0) < 1e-9 and m["leg_track_rms"] == m["leg_track_rms"]:
            baseline = m["leg_track_rms"] if baseline is None else min(
                baseline, m["leg_track_rms"])

    ok_kps, bad_kps = set(), set()
    for kp, fname, m in rows:
        why = []
        if not m["have_status"]:
            why.append("no ~/status (cannot claim survival)")
        if m["degraded"]:
            why.append("DEGRADED")
        if m["tilt_max_deg"] > TILT_LIMIT_DEG:
            why.append("tilt {:.0f}deg".format(m["tilt_max_deg"]))
        if baseline and m["leg_track_rms"] == m["leg_track_rms"] \
                and m["leg_track_rms"] > TRACKING_FACTOR * baseline:
            why.append("tracking {:.2f}x baseline".format(m["leg_track_rms"] / baseline))
        if m["tau_frac_max"] >= 1.0:
            why.append("torque at rating on {}".format(m["tau_worst_joint"]))
        if why:
            verdict = "FAIL: " + "; ".join(why)
        elif m["ended_running"]:
            # The recording stopped while the stack was still RUNNING, so this
            # point survived AT LEAST this long -- it is a lower bound, not a
            # measurement of when it would have stopped.
            verdict = "ok (>= this long; window ended while RUNNING)"
        else:
            verdict = "ok"
        (ok_kps if not why else bad_kps).add(kp)
        print("{:>6.2f}  {:>8.1f}  {:>9.1f}  {:>9.4f}  {:>8.0f}%  {:>9.2f}  {}".format(
            kp, m["running_s"], m["tilt_max_deg"], m["leg_track_rms"],
            100 * m["tau_frac_max"], m["chatter_hz"], verdict))
        if m["tau_all_zero"]:
            print("{:>6}  WARN tau_est is all zero -- some firmware leaves it "
                  "unset. Torque above is not a measurement.".format(""))

    if baseline is None:
        print("\nWARN  no kp=1.0 point, so there is no baseline and the tracking half"
              "\n      of the stability-domain definition cannot be applied. Record"
              "\n      the trained gain first next time.")

    good = sorted(ok_kps - bad_kps)
    print("\n=== stability domain (real robot) ===")
    print("  definition, fixed before measuring: no DEGRADED, tilt <= {:.0f} deg,"
          .format(TILT_LIMIT_DEG))
    print("  leg tracking rms <= {:.1f}x the kp=1.0 point, torque below rating"
          .format(TRACKING_FACTOR))
    if not good:
        print("  no point satisfies it.")
    else:
        lo, hi = min(good), max(good)
        print("  satisfied at: " + ", ".join("{:.2f}".format(k) for k in good))
        print("  interval    : {:.2f} .. {:.2f}  (width {:.2f})".format(lo, hi, hi - lo))
        # An edge is only a result if a point beyond it was tried and failed.
        tested = sorted(points)
        for side, edge in (("lower", lo), ("upper", hi)):
            beyond = [k for k in tested if (k < edge if side == "lower" else k > edge)]
            if beyond and any(k in bad_kps for k in beyond):
                print("  {} edge {:.2f}: BOUNDED -- {:.2f} was tried and failed"
                      .format(side, edge,
                              min(beyond) if side == "upper" else max(beyond)))
            else:
                print("  {} edge {:.2f}: NOT bounded -- nothing beyond it was tried, so"
                      " this is where testing stopped, not where the robot fails"
                      .format(side, edge))

    if sim_path:
        overlay(sim_path, good)
    else:
        print("\n  pass --sim <results.json> to state the sim2real gap (FR-R4): the"
              "\n  offset between the simulated curve and these points IS the gap.")
    return 0


def overlay(sim_path, good_real):
    p = pathlib.Path(sim_path)
    if not p.is_file():
        print("\nWARN  --sim {}: not found. The sim sweep "
              "(scripts/sweep_gain_robustness.py) has not been run yet.".format(sim_path))
        return
    try:
        sim = json.loads(p.read_text())
    except ValueError as exc:
        print("\nWARN  --sim {}: not valid JSON ({})".format(sim_path, exc))
        return
    # Accept either {"stable": [..]} or {"points": [{"kp_scale":..,"stable":bool}]}
    stable = sim.get("stable")
    if stable is None and isinstance(sim.get("points"), list):
        stable = [pt.get("kp_scale") for pt in sim["points"] if pt.get("stable")]
    if not stable:
        print("\nWARN  --sim {}: no stable-point list found; expected 'stable' or "
              "'points'.".format(sim_path))
        return
    slo, shi = min(stable), max(stable)
    print("\n=== sim2real gap (FR-R4) ===")
    print("  sim  stable {:.2f} .. {:.2f}  (width {:.2f})".format(slo, shi, shi - slo))
    if good_real:
        rlo, rhi = min(good_real), max(good_real)
        print("  real stable {:.2f} .. {:.2f}  (width {:.2f})".format(rlo, rhi, rhi - rlo))
        print("  gap  lower edge {:+.2f}, upper edge {:+.2f}".format(rlo - slo, rhi - shi))
        print("  read as: the robot tolerates {:.0%} of the gain error the simulator"
              " says it should".format((rhi - rlo) / max(1e-9, shi - slo)))
        print("  NOTE the real interval is built from a handful of points, so this is"
              "\n       a statement about the TREND agreeing, not a precise gap value.")
    else:
        print("  real: no point satisfied the definition.")


def show_plan(points):
    print(__doc__.split("\n\n")[0])
    print("\n=== point order ===")
    print("1.0 first (it is the trained controller and the baseline), then outward")
    print("alternately, so a session cut short still brackets the baseline:\n")
    print("  " + "  ".join("{:.1f}".format(k) for k in points))
    print("\n=== per point, two terminals ===")
    print("""
  # terminal 1 -- the stack. ros2 launch HOLDS this terminal: do not type here,
  # and do not Ctrl-C it to run a command, or the whole stack goes down.
  ros2 launch r1_hw_bridge r1_stack.launch.py \\
      engine:=$R1_DEPLOY_ROOT/artifacts/policy_fp32.plan iface:=eth10 \\
      enable_output:=true kp_scale:=<KP> kd_scale:=1.0

  # terminal 2 -- source env.sh here too, then record. The note is not optional:
  # without kp= the collector refuses the file.
  python3 $R1_DEPLOY_ROOT/tools/probe_cpp/walk_metrics.py --seconds 30 \\
      --save ~/orin_commissioning/gain_sweep/kp<KP>.json \\
      --note "kp=<KP> gantry slack, vx 0.4 commanded"
""")
    print("=== stop immediately if ===")
    print("  * any new noise from the joints, or visible tremor")
    print("  * the torso tilts past ~20 deg")
    print("  * the bridge prints DEGRADED")
    print("  That point is the real edge. Record what you have and do NOT push")
    print("  further out -- a failed point that was recorded is data; a fall is not.")
    print("\n  kp_scale below 1.0 makes the legs SOFTER. Hang the gantry.")
    print("  Above ~1.2 can ring. Listen before you look.")
    print("\n=== then ===")
    print("  python3 gain_sweep_real.py --collect ~/orin_commissioning/gain_sweep")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true", help="print the run checklist")
    g.add_argument("--collect", metavar="DIR", help="reduce the saved recordings")
    ap.add_argument("--sim", metavar="JSON", help="sim sweep results, for the gap line")
    ap.add_argument("--points", help="comma-separated gains, overrides the default order")
    args = ap.parse_args()

    if args.plan:
        pts = ([float(x) for x in args.points.split(",")] if args.points else PLAN_ORDER)
        return show_plan(pts)
    return collect(args.collect, args.sim)


if __name__ == "__main__":
    sys.exit(main())
