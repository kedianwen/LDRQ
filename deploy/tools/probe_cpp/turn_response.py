#!/usr/bin/env python3
"""How fast does the robot actually turn for a given wz command? (stage B)

Found 2026-09-29 in the PG-2 recording: 80 s of vx=0.1, wz=0.15, untethered, and the
robot turned 12 degrees in total -- a measured yaw rate of 0.00 rad/s -- while the
same deployed policy in simulation turns at 0.98-1.06 of every command from 0.15 to
0.5 rad/s. So there is a deadband somewhere on the real side, and mission_ctl's
closed-loop turn tapers down to 0.35 x 0.4 = 0.14 rad/s near its target, i.e. into it
(two of stage A's four turn timeouts stalled at 170 and 177 deg).

This tool measures the response curve so the turn controller can be configured
above the deadband. Nothing here commands the robot directly: --record runs
mission_ctl with OPEN-LOOP turns (`turn left 6s@0.25` is a constant wz for 6 s) while
walk_metrics records, and the analysis reads any walk_metrics recording -- including
ones made for other purposes, like the PG-2 run.

  # print the mission script and the time it takes (no robot needed)
  python3 turn_response.py --plan
  # on the robot, turning on the spot; a slack gantry may stay attached, but start with
  # the feet under the gantry point -- once the rope pulls, those segments are the rig
  python3 turn_response.py --record
  # analyze one or more recordings (any walk_metrics json)
  python3 turn_response.py --collect ~/orin_commissioning/turn/*.json

Left and right alternate, so the robot ends facing roughly where it started and a
left/right asymmetry shows up in the table.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Default grid: from below the suspected deadband to the trained limit (0.5).
WZ_GRID = [0.15, 0.25, 0.35, 0.5]
SEG_S = 6.0          # per direction per rate
PAUSE_S = 2.0        # stand between segments
SKIP_S = 1.5         # discard the ramp + transient at the start of each segment
# A segment is DISTURBED if the torso tilts past this, or the robot yaws the wrong way
# faster than WRONG_WAY for any half second. On the spot, undisturbed turning stays
# under ~5.5 deg; on 2026-09-29, 5 of 24 segments had a sudden 10-20 deg lurch with a
# 0.4-0.55 rad/s yaw kick, at several rates. They are listed but not averaged: a
# stumble is not the response to the command (it made 0.15 look like it turned backwards).
DISTURB_TILT_DEG = 8.0
WRONG_WAY = 0.1

# Simulation reference, same deployed policy, nominal conditions, 16 envs per command,
# measured 2026-09-29 (yaw over 27 s of steady command): ratio measured / commanded.
SIM_RATIO = {(0.1, 0.15): 1.03, (0.0, 0.15): 0.98, (0.3, 0.15): 1.06, (0.1, 0.3): 1.03,
             (0.0, 0.4): 0.99, (0.3, 0.4): 1.03, (0.1, 0.5): 1.01}


def quat_to_yaw(q):
    w, x, y, z = q[:4]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def build_script(grid, vx=0.0, seg_s=SEG_S, pause_s=PAUSE_S):
    parts = []
    for wz in grid:
        for side in ("left", "right"):
            if vx > 0:
                # an arc: walk and turn together is not a mission_ctl primitive, so
                # arcs are recorded with ros2 topic pub (see the stage B guide)
                raise SystemExit("[refuse] --vx > 0 needs an arc primitive; record arcs "
                                 "with ros2 topic pub and analyze them with --collect")
            parts.append("turn {} {:g}s@{:g}".format(side, seg_s, wz))
            parts.append("stand {:g}s".format(pause_s))
    return "; ".join(parts)


def segments(blob, skip_s=SKIP_S):
    """Constant-command stretches with a non-zero wz, from the command the policy saw."""
    obs = blob.get("obs", [])
    out, cur = [], None
    for t, v in obs:
        if len(v) < 9:
            continue
        key = (round(v[6], 3), round(v[8], 3))
        if cur is None or key != cur[0]:
            if cur is not None:
                out.append(cur)
            cur = [key, t, t]
        else:
            cur[2] = t
    if cur is not None:
        out.append(cur)
    # keep turning segments long enough to measure; ramps make many tiny ones
    return [(k, a + skip_s, b) for k, a, b in out if abs(k[1]) > 1e-3 and b - a > skip_s + 1.0]


def yaw_rate(blob, t0, t1):
    imu = [(t, v) for t, v in blob.get("imu", []) if t0 <= t <= t1 and len(v) >= 7]
    if len(imu) < 10:
        return None, None
    acc, prev = 0.0, None
    for _t, v in imu:
        y = quat_to_yaw(v)
        if prev is not None:
            acc += (y - prev + math.pi) % (2 * math.pi) - math.pi
        prev = y
    rate = acc / (imu[-1][0] - imu[0][0])
    gyro = sum(v[6] for _t, v in imu) / len(imu)
    return rate, gyro


def tilt_deg(q):
    """Angle between the torso's z axis and vertical."""
    _w, x, y, _z = q[:4]
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def disturbance(blob, t0, t1, wz, step=0.5):
    """(max tilt, list of offsets of half-seconds spent yawing the wrong way)."""
    tilts = [tilt_deg(v) for t, v in blob.get("imu", []) if t0 <= t <= t1 and len(v) >= 4]
    wrong, s = [], t0
    while s + step <= t1 + 1e-6:
        r, _g = yaw_rate(blob, s, s + step)
        if r is not None and r * wz < 0 and abs(r) > WRONG_WAY:
            wrong.append(s - t0)
        s += step
    return (max(tilts) if tilts else 0.0), wrong


def collect(paths):
    rows = []
    for p in paths:
        blob = json.loads(pathlib.Path(p).read_text())
        for (vx, wz), a, b in segments(blob):
            rate, gyro = yaw_rate(blob, a, b)
            if rate is None:
                continue
            tmax, wrong = disturbance(blob, a, b, wz)
            bad = tmax > DISTURB_TILT_DEG or bool(wrong)
            rows.append((pathlib.Path(p).name, vx, wz, b - a, rate, gyro, tmax, bad))
    if not rows:
        print("no turning segments found (no non-zero wz held for > {:.1f} s)".format(SKIP_S + 1))
        return 1
    print("measured yaw rate per constant-command segment (first {:.1f} s skipped)".format(SKIP_S))
    print("{:<28} {:>5} {:>6} {:>6} {:>9} {:>9} {:>7} {:>6} {:>5}".format(
        "recording", "vx", "wz cmd", "secs", "yaw rate", "gyro z", "ratio", "sim", "tilt"))
    for name, vx, wz, dur, rate, gyro, tmax, bad in rows:
        sim = SIM_RATIO.get((round(vx, 2), round(abs(wz), 2)))
        print("{:<28} {:>5.2f} {:>+6.2f} {:>6.1f} {:>+9.3f} {:>+9.3f} {:>7.2f} {:>6} {:>5.1f}{}".format(
            name[:28], vx, wz, dur, rate, gyro, rate / wz, "{:.2f}".format(sim) if sim else "-",
            tmax, "  DISTURBED" if bad else ""))
    n_bad = sum(1 for r in rows if r[7])
    if n_bad:
        print("\nDISTURBED = tilt > {:.0f} deg or a half second yawing the wrong way at > {:.1f} "
              "rad/s: a lurch, not the response to the command. {} of {} segments; left out "
              "below.".format(DISTURB_TILT_DEG, WRONG_WAY, n_bad, len(rows)))
    # the number the turn controller needs: the smallest command that turns properly
    by_wz = {}
    for _n, _vx, wz, _d, rate, _g, _t, bad in rows:
        if not bad:
            by_wz.setdefault(abs(wz), []).append(rate / wz)
    print("\nby |wz| (mean ratio over undisturbed directions and recordings):")
    ok = []
    for wz in sorted(by_wz):
        r = sum(by_wz[wz]) / len(by_wz[wz])
        print("  {:.2f} rad/s  ->  {:.2f} of the command  (n={}, {:.2f}-{:.2f}){}".format(
            wz, r, len(by_wz[wz]), min(by_wz[wz]), max(by_wz[wz]),
            "" if r >= 0.7 else "   <- below 0.7: treat as not turning"))
        if r >= 0.7:
            ok.append(wz)
    if ok:
        lowest = min(by_wz)
        print("\nsmallest |wz| that turns at >= 0.7 of the command: {:.2f} rad/s{}".format(
            min(ok), "  (the lowest rate tested: the edge is at or below it)"
            if min(ok) == lowest else ""))
        print("  -> mission_ctl/config/mission.yaml:  turn_min_wz: {:.2f}".format(min(ok)))
        print("     (the closed-loop turn will never command less than this near its target)")
    else:
        print("\nno tested |wz| turns at >= 0.7 of the command.")
    return 0


def record(grid, seg_s, pause_s, out_dir, note, assume_yes):
    script = build_script(grid, seg_s=seg_s, pause_s=pause_s)
    total = len(grid) * 2 * (seg_s + pause_s)
    mission = HERE.parents[2] / "mission_ctl" / "r1_mission_cli.py"
    if not mission.is_file():
        print("[refuse] mission_ctl not found at {}".format(mission))
        return 2
    out_dir = pathlib.Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    save = out_dir / "turn_{}.json".format(time.strftime("%m%d_%H%M%S"))
    print("mission: " + script)
    print("about to record {:.0f} s -> {}".format(total + 12, save))
    print("Turning on the spot: start under the gantry point (rope slack), ~1.5 m clear.")
    if not assume_yes:
        try:
            if input("go? [y/N] ").strip().lower() not in ("y", "yes"):
                print("not recorded.")
                return 2
        except EOFError:
            print("not recorded (no tty; pass --yes).")
            return 2
    wm = subprocess.Popen([sys.executable, str(HERE / "walk_metrics.py"), "--seconds",
                           str(total + 12), "--save", str(save), "--note",
                           "turn_response grid={} {}".format(",".join(map(str, grid)), note)])
    try:
        time.sleep(1.5)
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        rc = subprocess.call([sys.executable, str(mission), "run", script, "--yes"], env=env)
        wm.wait()
    except KeyboardInterrupt:
        wm.terminate()
        print("\ninterrupted: the mission process exited; the bridge deadman stops the robot.")
        return 1
    print("\nmission exit {}  ->  {}".format(rc, save))
    return collect([save]) if save.is_file() else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true", help="print the mission script")
    g.add_argument("--record", action="store_true", help="record the response on the robot")
    g.add_argument("--collect", nargs="+", metavar="JSON", help="analyze recordings")
    ap.add_argument("--grid", default=",".join(map(str, WZ_GRID)), help="wz values, rad/s")
    ap.add_argument("--seconds", type=float, default=SEG_S, help="per direction per rate")
    ap.add_argument("--pause", type=float, default=PAUSE_S)
    ap.add_argument("--out", default="~/orin_commissioning/turn")
    ap.add_argument("--note", default="on the spot, gantry slack")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    grid = [float(x) for x in a.grid.split(",")]
    if a.plan:
        s = build_script(grid, seg_s=a.seconds, pause_s=a.pause)
        print(s)
        print("\n{} segments, ~{:.0f} s. Check it compiles: python3 {} run \"<script>\" --dry-run".format(
            2 * len(grid), len(grid) * 2 * (a.seconds + a.pause),
            HERE.parents[2] / "mission_ctl" / "r1_mission_cli.py"))
        return 0
    if a.record:
        return record(grid, a.seconds, a.pause, a.out, a.note, a.yes)
    return collect(a.collect)


if __name__ == "__main__":
    sys.exit(main())
