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

  # one point, one command (terminal 2, with the stack already up at that gain)
  python3 gain_sweep_real.py --record 0.8

  # after the session, from the saved recordings
  python3 gain_sweep_real.py --collect ~/orin_commissioning/gain_sweep
  python3 gain_sweep_real.py --collect ~/orin_commissioning/gain_sweep \
      --sim <repo>/outputs/gain_sweep/<date>/results.json

The gain label is the weakest link in the whole sweep: ~/status does not carry
kp_scale, and kp_scale is read once at start-up, so a point recorded after a
restart the operator forgot to do carries the PREVIOUS gain under the new label.
--record therefore asks the running bridge for its kp_scale / kd_scale /
enable_output through the parameter service and refuses to record if they
disagree with the label, then writes the verified values into the note.
A recording whose note has no `kp=<value>` is refused by --collect rather than
guessed at; one written by --record additionally says `verified=param`.

Learned from the first real session (2026-09-28, ten recordings, all on a slack
gantry):
  * Turns under the gantry ran 1.1-1.7x slower than a perfectly tracking robot,
    and the second straight walk yawed BACK by up to 60 deg -- rope torsion, not
    the gain (the two kp=1.10 repeats disagreed: one finished its turn, one timed
    out). The recording window was sized for the ideal plan, so six points were
    cut off mid-turn and four ended in a turn timeout followed by standing.
    --record now sizes the window for the worst case and saves mission_ctl's
    report beside the recording; --collect scores only the commanded window, so
    standing after an abort does not flatter a point.
  * The policy node publishes no ~/action, so the chatter column read 0.00 as if
    measured. --collect now takes the last action from the ~/obs frame instead.

A point the operator stopped by the --plan checklist (new joint noise, tremor,
tilt, DEGRADED) may have no usable recording. Write it into operator_log.txt in
the recordings directory, one line per stop:

    1.60 FAIL audible joint noise, e-stopped    # kp, FAIL, what was observed

Only FAIL is accepted: an operator's word can close an edge or exclude a gain,
never put one inside the domain -- that still takes a recording.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import walk_metrics as W   # noqa: E402  -- helpers, and the gains it already parses

# Run 1.0 first (it is the trained controller and the baseline every other point
# is judged against), then step outward alternately so that if the session is cut
# short the surviving points still bracket the baseline.
PLAN_ORDER = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6, 0.5]

# The command sequence every point is walked with -- on the robot AND in the sim
# sweep. If the two sides are commanded differently, the offset between the
# simulated curve and the real points is not a sim2real gap, it is a difference in
# what was asked for. Out and back in a ~1.5 m lane, so it fits under a gantry,
# and it exercises both walking and turning. The sim sweep must import this
# string (and mission_ctl's Executor, which is pure Python) rather than retype it.
SWEEP_SCRIPT = "walk 5s@0.3; turn left 180@0.4; walk 5s@0.3; turn left 180@0.4"

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


def frame_offsets(terms):
    """{term: (start, width)} inside ONE 85-float ~/obs frame, in spec order."""
    out, at = {}, 0
    for name, spec in terms.items():
        out[name] = (at, int(spec["width"]))
        at += int(spec["width"])
    return out


def command_window(blob, off):
    """Where the commanded motion actually is, read from the velocity command the
    policy was given (the obs frame), not from wall-clock guesses.

    Returns (t_first, t_last, still_moving_at_end, segments) or None, where
    segments counts walk/turn alternations -- compared against the script it says
    whether the whole sequence was walked."""
    obs = blob.get("obs", [])
    if "velocity_commands" not in off or not obs:
        return None
    c0, _w = off["velocity_commands"]
    moving = [(t, v[c0], v[c0 + 2]) for t, v in obs if len(v) > c0 + 2
              and (abs(v[c0]) > 1e-3 or abs(v[c0 + 2]) > 1e-3)]
    if not moving:
        return None
    segs, last = 0, None
    for _t, vx, _wz in moving:
        kind = "walk" if vx > 0.01 else "turn"
        if kind != last:
            segs, last = segs + 1, kind
    end_v = obs[-1][1]
    still = abs(end_v[c0]) > 1e-3 or abs(end_v[c0 + 2]) > 1e-3
    return moving[0][0], moving[-1][0], still, segs


def windowed(blob, t0, t1):
    """The same recording, cut to [t0, t1]. ~/status keeps everything up to t1,
    because survival is judged on the whole run up to that point."""
    out = dict(blob)
    out["t_start"], out["t_end"] = t0, t1
    for k in ("jpos", "jvel", "tau", "cmd", "imu", "obs", "action"):
        out[k] = [x for x in blob.get(k, []) if t0 <= x[0] <= t1]
    out["status"] = [x for x in blob.get("status", []) if x[0] <= t1]
    return out


def script_segments(script):
    """How many walk/turn alternations the script should produce."""
    kinds, last = 0, None
    for st in _mission_plan().parse_script(script):
        if st["op"] not in ("walk", "turn"):
            continue
        if st["op"] != last:
            kinds, last = kinds + 1, st["op"]
    return kinds


def parse_script_note(note):
    s = str(note)
    i = s.find("script=[")
    if i < 0:
        return None
    j = s.find("]", i)
    return s[i + len("script=["):j] if j > i else None


def mission_outcome(path):
    """mission_ctl's own verdict, from the sidecar --record saves (newer runs)."""
    side = path.with_suffix(".mission.txt")
    if not side.is_file():
        return None
    text = side.read_text()
    i = text.find('"final_state"')
    if i < 0:
        return "no report (mission exited early?)"
    state = text[i:].split(":", 1)[1].split(",", 1)[0].strip().strip('"')
    j = text.find('"abort_reason"')
    reason = text[j:].split(":", 1)[1].split("\n", 1)[0].strip().rstrip(",").strip('"') \
        if j >= 0 else "null"
    return state if reason in ("null", "None") else "{}: {}".format(state, reason)


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
    srt = sorted(tilts)
    out["tilt_p99_deg"] = srt[min(len(srt) - 1, int(0.99 * len(srt)))] if srt \
        else float("nan")
    imu_hz = len(tilts) / max(1e-9, t1 - t0)
    out["tilt_over_s"] = sum(1 for x in tilts if x > TILT_LIMIT_DEG) / max(1e-9, imu_hz)

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


def read_operator_log(directory):
    """[(kp, reason)] from <directory>/operator_log.txt; FAIL lines only."""
    path = pathlib.Path(directory) / "operator_log.txt"
    if not path.is_file():
        return []
    out = []
    for n, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(None, 2)
        try:
            kp = float(parts[0])
        except (ValueError, IndexError):
            raise SystemExit("[fail] operator_log.txt:{}: expected '<kp> FAIL <what was "
                             "observed>', got {!r}".format(n, raw))
        if len(parts) < 2 or parts[1].upper() != "FAIL":
            raise SystemExit("[fail] operator_log.txt:{}: only FAIL is accepted -- a gain "
                             "goes INTO the domain only on a recording".format(n))
        out.append((kp, parts[2] if len(parts) > 2 else "(no reason given)"))
    return out


def load_recordings(directory, quiet=False):
    """{kp: [(file name, metrics)]} and [(file name, why refused)] -- the reduction
    --collect prints, and what plotting reads (scripts/plot_gain_sweep.py)."""
    files = sorted(pathlib.Path(directory).glob("*.json"))
    if not files:
        raise SystemExit("[fail] no *.json recordings in " + str(directory))

    defaults, terms, _an = W.load_spec()
    kp_gains, names, groups = W.parse_hpp_array("kKp")
    off = frame_offsets(terms)

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
        if kp is not None and not quiet and "verified=param" not in str(blob.get("note", "")):
            print("NOTE     {}: kp={:.2f} is from a hand-typed note, not read back from"
                  " the bridge".format(f.name, kp))
        if kp is None:
            refused.append((f.name, "no 'kp=<value>' in its note -- refusing to guess "
                                    "which gain this point was recorded at"))
            continue
        points.setdefault(kp, []).append((f.name, reduce_recording(
            f, blob, off, defaults, kp_gains, names, groups)))
    return points, refused


def collect(directory, sim_path=None):
    points, refused = load_recordings(directory)
    for name, why in refused:
        print("REFUSED  {}: {}".format(name, why))
    if refused:
        print()
    if not points:
        raise SystemExit("[fail] no usable points")
    op_log = read_operator_log(directory)

    print("=== per point ({} recordings at {} gain(s)) ===".format(
        sum(len(v) for v in points.values()), len(points)))
    print("scored over the COMMANDED window only (first to last non-zero velocity"
          "\ncommand in ~/obs); 'stand' is the median tilt before the first command.")
    print("{:>6}  {:>6}  {:>6}  {:>8}  {:>6}  {:>8}  {:>6}  {:>7}  {}".format(
        "kp", "run s", "stand", "tilt max", "p99", "leg rms", "tau %", "chat Hz",
        "verdict"))

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
    tally = {}                       # kp -> [recordings, failed]
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
            if len(why) == 1 and why[0].startswith("tilt") and m["tilt_over_s"] < 0.2:
                # Still a FAIL -- the definition was fixed before measuring and is
                # not re-drawn after. But say how thin the evidence is.
                verdict += " (marginal: {:.2f} s above the limit, p99 {:.1f} deg)".format(
                    m["tilt_over_s"], m["tilt_p99_deg"])
        elif m["ended_running"]:
            # The recording stopped while the stack was still RUNNING, so this
            # point survived AT LEAST this long -- it is a lower bound, not a
            # measurement of when it would have stopped.
            verdict = "ok (>= this long; window ended while RUNNING)"
        else:
            verdict = "ok"
        (ok_kps if not why else bad_kps).add(kp)
        t = tally.setdefault(kp, [0, 0])
        t[0] += 1
        t[1] += 1 if why else 0
        print("{:>6.2f}  {:>6.1f}  {:>6.1f}  {:>8.1f}  {:>6.1f}  {:>8.4f}  {:>5.0f}%  {:>6.2f}{}  {}"
              .format(kp, m["running_s"], m["stand_tilt_deg"], m["tilt_max_deg"],
                      m["tilt_p99_deg"], m["leg_track_rms"], 100 * m["tau_frac_max"],
                      m["chatter_hz"], "*" if m["chatter_src"] == "obs" else " ",
                      verdict))
        print("{:>6}  plan: {}".format("", m["outcome"]))
        if m["tau_all_zero"]:
            print("{:>6}  WARN tau_est is all zero -- some firmware leaves it "
                  "unset. Torque above is not a measurement.".format(""))

    if any(m["chatter_src"] == "obs" for _k, _f, m in rows):
        print("\n  * chatter from the last action inside the ~/obs frame: the policy node"
              "\n    publishes no ~/action. Same signal, one 50 Hz sample later.")
    short = [(k, m["outcome"]) for k, _f, m in rows if not m["outcome"].startswith(
        ("complete", "DONE"))]
    if short:
        print("\nWARN  {} of {} recordings did not walk the whole sequence, so those"
              "\n      points were not asked the same thing as the others (or as the sim)."
              "\n      Stability numbers inside the window still stand; turn times do not."
              "\n      On a gantry, a turn that times out is rope torsion until shown"
              "\n      otherwise -- untwist the rope between points.".format(
                  len(short), len(rows)))

    if baseline is None:
        print("\nWARN  no kp=1.0 point, so there is no baseline and the tracking half"
              "\n      of the stability-domain definition cannot be applied. Record"
              "\n      the trained gain first next time.")

    if op_log:
        print("\n=== stopped by the operator (--plan checklist; no usable recording) ===")
        for kp, why in sorted(op_log, reverse=True):
            print("{:>6.2f}  FAIL (operator): {}".format(kp, why))
            bad_kps.add(kp)
            t = tally.setdefault(kp, [0, 0])
            t[0] += 1
            t[1] += 1
    op_kps = {kp for kp, _ in op_log}

    good = sorted(ok_kps - bad_kps)
    mixed = sorted(ok_kps & bad_kps)

    def failed_of(k):
        s = "{} of {} failed".format(tally[k][1], tally[k][0])
        return s + (" (operator stop)" if k in op_kps else "")

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
        if mixed:
            # A gain is in the domain only if EVERY recording at it passes -- fixed
            # before measuring, so a repeat can confirm a failure but never wash it out.
            print("  mixed       : " + ", ".join("{:.2f} ({})".format(k, failed_of(k))
                                                  for k in mixed)
                  + " -- excluded, and the edge next to it is marginal")
        print("  interval    : {:.2f} .. {:.2f}  (width {:.2f})".format(lo, hi, hi - lo))
        # An edge is only a result if a point beyond it was tried and failed.
        tested = sorted(set(points) | op_kps)
        for side, edge in (("lower", lo), ("upper", hi)):
            beyond = [k for k in tested if (k < edge if side == "lower" else k > edge)]
            if beyond and any(k in bad_kps for k in beyond):
                nxt = min(beyond) if side == "upper" else max(beyond)
                print("  {} edge {:.2f}: BOUNDED -- {:.2f} was tried: {}"
                      .format(side, edge, nxt, failed_of(nxt)))
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


def reduce_recording(path, blob, off, defaults, kp_gains, names, groups):
    """One recording -> the numbers, scored over the commanded window."""
    win = command_window(blob, off)
    if win is None:
        cut, outcome = blob, "no motion was commanded in this recording"
        stand = []
    else:
        cut = windowed(blob, win[0], win[1])
        stand = [tilt_deg(v[7:10]) for t, v in blob.get("imu", [])
                 if t < win[0] and len(v) >= 10]
        outcome = mission_outcome(path)
        if outcome is None:
            script = parse_script_note(blob.get("note", ""))
            want = None
            if script:
                try:
                    want = script_segments(script)
                except Exception:        # an unparseable note is not fatal here
                    want = None
            if win[2]:
                outcome = "CUT OFF: still commanding when the recording ended"
            elif want is not None and win[3] < want:
                outcome = ("ended early after {} of {} walk/turn segments (a turn "
                           "timeout or an abort; no mission report saved)"
                           .format(win[3], want))
            else:
                outcome = "complete (inferred from the command trace)"
    src = "action"
    if not cut.get("action") and "actions" in off:
        a0, aw = off["actions"]
        cut = dict(cut)
        cut["action"] = [(t, v[a0:a0 + aw]) for t, v in cut.get("obs", [])
                         if len(v) >= a0 + aw]
        src = "obs"
    m = reduce_point(cut, defaults, kp_gains, names, groups)
    srt = sorted(stand)
    m["stand_tilt_deg"] = srt[len(srt) // 2] if srt else float("nan")
    m["outcome"] = outcome
    m["chatter_src"] = src
    return m


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
    # A sim edge at the end of the simulated grid was not reached, only stopped at:
    # the gap on that side is then a bound, and the ratio below with it.
    tried = [pt.get("kp_scale") for pt in sim.get("points", [])] or list(stable)
    lo_open, hi_open = slo <= min(tried), shi >= max(tried)
    print("\n=== sim2real gap (FR-R4) ===")
    print("  sim  stable {}{:.2f} .. {:.2f}{}  (width {}{:.2f})".format(
        "<=" if lo_open else "", slo, shi, "+ (grid ends here)" if hi_open else "",
        ">=" if (lo_open or hi_open) else "", shi - slo))
    if good_real:
        rlo, rhi = min(good_real), max(good_real)
        print("  real stable {:.2f} .. {:.2f}  (width {:.2f})".format(rlo, rhi, rhi - rlo))
        print("  gap  lower edge {}{:+.2f}, upper edge {}{:+.2f}".format(
            ">=" if lo_open else "", rlo - slo, "<=" if hi_open else "", rhi - shi))
        print("  read as: the robot tolerates {}{:.0%} of the gain error the simulator"
              " says it should".format("at most " if (lo_open or hi_open) else "",
                                       (rhi - rlo) / max(1e-9, shi - slo)))
        print("  NOTE the real interval is built from a handful of points, so this is"
              "\n       a statement about the TREND agreeing, not a precise gap value.")
    else:
        print("  real: no point satisfied the definition.")


def query_bridge_params(names, node_name="/r1_hw_bridge", timeout=5.0):
    """Ask the running bridge for its parameters. rcl_interfaces GetParameters
    exists on foxy, unlike the newer AsyncParameterClient."""
    import rclpy
    from rcl_interfaces.srv import GetParameters
    rclpy.init()
    node = rclpy.create_node("gain_sweep_param_probe")
    try:
        cli = node.create_client(GetParameters, node_name + "/get_parameters")
        if not cli.wait_for_service(timeout_sec=timeout):
            return None, "no {}/get_parameters service -- is the stack up, and was " \
                         "this terminal set up with the same ROS_DOMAIN_ID / " \
                         "ROS_LOCALHOST_ONLY / RMW as the stack?".format(node_name)
        req = GetParameters.Request()
        req.names = list(names)
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
        if not fut.done() or fut.result() is None:
            return None, "parameter query timed out"
        out = {}
        for name, v in zip(names, fut.result().values):
            # type 1 bool, 2 int, 3 double (rcl_interfaces/ParameterType)
            out[name] = {1: v.bool_value, 2: v.integer_value, 3: v.double_value}.get(
                v.type, None)
        return out, None
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _mission_plan():
    mdir = HERE.parents[2] / "mission_ctl"
    if str(mdir) not in sys.path:
        sys.path.insert(0, str(mdir))
    from r1_mission import plan as MP       # noqa: E402
    return MP


def script_duration(script):
    """Validate the script with mission_ctl's own compiler, before anything talks
    to the robot, and return how long to RECORD: long enough for the worst case
    mission_ctl allows -- every closed-loop turn running to its timeout -- not
    the ideal. Sized for the ideal, the 2026-09-28 session cut six of ten points
    off mid-turn. Standing after the mission ends costs nothing: --collect scores
    only the commanded window."""
    MP = _mission_plan()
    from r1_mission import node as MN       # noqa: E402  (no rclpy at import)
    env = MP.Envelope.from_bridge_yaml(
        HERE.parents[1] / "ros2_ws/src/r1_hw_bridge/config/bridge.yaml")
    cfg = HERE.parents[2] / "mission_ctl" / "config" / "mission.yaml"
    lim = MN.limits_from_cfg(MN.read_flat_yaml(str(cfg))) if cfg.is_file() \
        else MP.Limits()
    prims, _tot_s, _tot_m = MP.compile_plan(MP.parse_script(script), env, lim)
    worst = 0.0
    for p in prims:
        if p.closed_loop:
            worst += (p.expected_s or p.duration_s or 0.0) * lim.turn_timeout_factor
        else:
            worst += (p.duration_s or 0.0) + lim.ramp_s
    return worst + lim.settle_s + 3.0


def record(kp, script, out_dir, extra_note, assume_yes, drive, seconds=None):
    import subprocess
    import time
    got, err = query_bridge_params(["kp_scale", "kd_scale", "enable_output"])
    if err:
        print("[refuse] " + err)
        return 2
    print("bridge reports: kp_scale={} kd_scale={} enable_output={}".format(
        got.get("kp_scale"), got.get("kd_scale"), got.get("enable_output")))
    if got.get("kp_scale") is None or abs(float(got["kp_scale"]) - kp) > 1e-6:
        print("[refuse] you asked to record kp={:.2f} but the running bridge has "
              "kp_scale={}.\n         kp_scale is read once at start-up: restart "
              "the stack with kp_scale:={:.2f} first.".format(kp, got.get("kp_scale"), kp))
        return 2
    if drive and not got.get("enable_output"):
        print("[refuse] enable_output is false: the robot would not move, and the "
              "point would record a standing robot under a walking label.")
        return 2

    out_dir = pathlib.Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%m%d_%H%M%S")
    save = out_dir / "kp{:.2f}_{}.json".format(kp, stamp)
    mission = HERE.parents[2] / "mission_ctl" / "r1_mission_cli.py"
    if drive and not mission.is_file():
        print("[refuse] mission_ctl not found at {} -- install the stage A package, "
              "or pass --no-drive and command the robot yourself.".format(mission))
        return 2
    if drive:
        try:
            seconds = script_duration(script)
        except Exception as exc:      # PlanError, or mission_ctl not importable
            print("[refuse] the sweep script does not compile: {}".format(exc))
            return 2
    elif seconds is None:
        seconds = 30.0
    note = "kp={:.2f} kd={} verified=param script=[{}] {}".format(
        kp, got.get("kd_scale"), script if drive else "none (operator drove)",
        extra_note).strip()

    print("\nabout to: record {:.0f} s -> {}".format(seconds + 2, save))
    if drive:
        print("          while mission_ctl runs: " + script)
    if not assume_yes:
        try:
            if input("go? [y/N] ").strip().lower() not in ("y", "yes"):
                print("not recorded.")
                return 2
        except EOFError:
            print("not recorded (no tty; pass --yes).")
            return 2

    wm = subprocess.Popen([sys.executable, str(HERE / "walk_metrics.py"),
                           "--seconds", str(seconds + 2), "--save", str(save),
                           "--note", note])
    rc_drive = 0
    side = save.with_suffix(".mission.txt")
    try:
        time.sleep(1.5)       # let the recorder subscribe before anything moves
        if drive:
            # mission_ctl's report goes beside the recording, so --collect can say
            # DONE / ABORTED and why, instead of inferring it from the trace.
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            with open(str(side), "w") as fh:
                proc = subprocess.Popen([sys.executable, str(mission), "run",
                                         script, "--yes"], stdout=subprocess.PIPE,
                                        universal_newlines=True, env=env)
                for line in proc.stdout:
                    sys.stdout.write(line)
                    fh.write(line)
                rc_drive = proc.wait()
            print("\nmission finished; the recorder runs to the end of its worst-case"
                  "\nwindow ({:.0f} s) -- only the commanded part is scored.".format(
                      seconds + 2))
        rc_wm = wm.wait()
    except KeyboardInterrupt:
        wm.terminate()
        print("\ninterrupted: the mission process exited, so the bridge deadman "
              "brings the robot to zero within 0.5 s.")
        return 1
    print("\nrecorder exit {}  mission exit {}  ->  {}".format(rc_wm, rc_drive, save))
    if rc_drive == 1:
        why = mission_outcome(save) or ""
        if "turn timed out" in why:
            print("mission ABORTED on a TURN TIMEOUT ({}).\nThat is not a stability "
                  "edge by itself. On a gantry, first untwist the rope (turn the "
                  "robot back by hand with output off) and repeat this point once."
                  .format(why))
        else:
            print("mission ABORTED ({}). That is a data point: keep the file, and do "
                  "NOT push the gain further out.".format(why or "see its report"))
    return 0 if rc_wm == 0 else 1


def show_plan(points):
    print(__doc__.split("\n\n")[0])
    print("\n=== point order ===")
    print("1.0 first (it is the trained controller and the baseline), then outward")
    print("alternately, so a session cut short still brackets the baseline:\n")
    print("  " + "  ".join("{:.1f}".format(k) for k in points))
    print("\n=== per point, two terminals ===")
    print(("""
  # terminal 1 -- the stack. ros2 launch HOLDS this terminal: do not type here,
  # and do not Ctrl-C it to run a command, or the whole stack goes down.
  ros2 launch r1_hw_bridge r1_stack.launch.py \\
      engine:=$R1_DEPLOY_ROOT/artifacts/policy_fp32.plan iface:=eth10 \\
      enable_output:=true kp_scale:=<KP> kd_scale:=1.0

  # terminal 2 -- source env.sh here too. One command per point: it reads the
  # gain back from the running bridge, refuses on mismatch, then records while
  # mission_ctl walks the fixed out-and-back sequence (~1.5 m lane, ~30 s):
  #   {}
  python3 $R1_DEPLOY_ROOT/tools/probe_cpp/gain_sweep_real.py --record <KP>
""").format(SWEEP_SCRIPT))
    print("=== stop immediately if ===")
    print("  * any new noise from the joints, or visible tremor")
    print("  * the torso tilts past ~20 deg")
    print("  * the bridge prints DEGRADED")
    print("  That point is the real edge. Record what you have and do NOT push")
    print("  further out -- a failed point that was recorded is data; a fall is not.")
    print("\n  kp_scale below 1.0 makes the legs SOFTER. Hang the gantry.")
    print("  Above ~1.2 can ring. Listen before you look.")
    print("\n  Every point turns +360 deg in total. Between points, with output off,")
    print("  turn the robot back by hand so the gantry rope is not twisted: on")
    print("  2026-09-28 a twisted rope slowed turns until they timed out, and yawed")
    print("  the robot 60 deg during a straight walk.")
    print("\n=== then ===")
    print("  python3 gain_sweep_real.py --collect ~/orin_commissioning/gain_sweep")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true", help="print the run checklist")
    g.add_argument("--collect", metavar="DIR", help="reduce the saved recordings")
    g.add_argument("--record", metavar="KP", type=float,
                   help="verify the running gain, then record one point")
    ap.add_argument("--script", default=SWEEP_SCRIPT,
                    help="--record: mission_ctl script (default: the shared sweep "
                         "sequence -- change it only together with the sim side)")
    ap.add_argument("--seconds", type=float, default=None,
                    help="--record --no-drive: recording length")
    ap.add_argument("--out", default="~/orin_commissioning/gain_sweep",
                    help="--record: where recordings go")
    ap.add_argument("--note", default="gantry slack", help="--record: extra note")
    ap.add_argument("--yes", action="store_true", help="--record: skip the prompt")
    ap.add_argument("--no-drive", action="store_true",
                    help="--record: do not command the robot (you drive it yourself)")
    ap.add_argument("--sim", metavar="JSON", help="sim sweep results, for the gap line")
    ap.add_argument("--points", help="comma-separated gains, overrides the default order")
    args = ap.parse_args()

    if args.plan:
        pts = ([float(x) for x in args.points.split(",")] if args.points else PLAN_ORDER)
        return show_plan(pts)
    if args.record is not None:
        return record(args.record, args.script, args.out, args.note,
                      args.yes, not args.no_drive, args.seconds)
    return collect(args.collect, args.sim)


if __name__ == "__main__":
    sys.exit(main())
