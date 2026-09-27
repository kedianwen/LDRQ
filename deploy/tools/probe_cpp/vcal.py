#!/usr/bin/env python3
"""Turn tape-measure + stopwatch readings into a calibrated walking speed.

Why this exists: the 425-dim observation has no base linear velocity term and the
robot publishes no odometry, so achieved speed cannot be recovered from anything
in the ROS graph. It has to be measured with a tape and a stopwatch, and then it
becomes the ONLY thing that turns "walk 10 m" into a duration.

This tool does three things the arithmetic alone does not:

  1. reports the SPREAD, so the error bar shipped with every distance estimate is
     measured rather than guessed;
  2. tests the assumption `mission_ctl` actually makes -- that achieved speed is
     PROPORTIONAL to commanded speed -- by fitting both a through-origin and a
     free-intercept line and comparing them. Sim already says the relationship
     bends at the top (0.022 m/s tracking error at 0.5 m/s, 0.236 at 1.0), so
     this is not a formality;
  3. emits the exact mission.yaml lines, so nobody retypes a number.

Readings file, one run per line (blank lines and # comments ignored):

    # vx_commanded  distance_m  seconds
    0.3   4.20   15.0
    0.3   4.15   15.1
    0.5   7.05   15.0

    python3 vcal.py readings.txt
    python3 vcal.py readings.txt --cruise 0.4     # where mission.yaml's v_cal sits
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys


def read_readings(path):
    rows = []
    for lineno, raw in enumerate(pathlib.Path(path).read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            raise SystemExit("[fail] {}:{}: expected 3 numbers "
                             "(vx_commanded distance_m seconds), got {!r}"
                             .format(path, lineno, line))
        try:
            vx, dist, secs = (float(x) for x in parts)
        except ValueError:
            raise SystemExit("[fail] {}:{}: not numbers: {!r}".format(path, lineno, line))
        if secs <= 0 or dist <= 0 or vx <= 0:
            raise SystemExit("[fail] {}:{}: all three must be > 0".format(path, lineno))
        rows.append((vx, dist, secs))
    if not rows:
        raise SystemExit("[fail] {}: no readings".format(path))
    return rows


def fit_through_origin(xs, ys):
    """y = k x. Least squares with the intercept pinned at zero, which is the
    model mission_ctl uses when it scales v_cal by vx / cruise_vx."""
    sxx = sum(x * x for x in xs)
    if sxx == 0:
        return 0.0
    return sum(x * y for x, y in zip(xs, ys)) / sxx


def fit_free(xs, ys):
    """y = a x + b."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return 0.0, my
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return a, my - a * mx


def rss(xs, ys, predict):
    return sum((y - predict(x)) ** 2 for x, y in zip(xs, ys))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("readings")
    ap.add_argument("--cruise", type=float, default=0.4,
                    help="mission.yaml's cruise_vx: v_cal is reported AT this "
                         "commanded speed (default 0.4)")
    ap.add_argument("--min-runs", type=int, default=3,
                    help="warn below this many runs per commanded speed")
    args = ap.parse_args()

    rows = read_readings(args.readings)
    by_speed = {}
    for vx, dist, secs in rows:
        by_speed.setdefault(vx, []).append(dist / secs)

    print("readings: {} runs at {} commanded speed(s)\n".format(len(rows), len(by_speed)))
    print("{:>10}  {:>4}  {:>10}  {:>9}  {:>9}  {:>9}".format(
        "commanded", "n", "achieved", "sd", "rel sd", "achieved/cmd"))
    xs, ys, rel_devs = [], [], []
    thin = []
    for vx in sorted(by_speed):
        vals = by_speed[vx]
        mean = statistics.fmean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
        rel = (sd / mean) if len(vals) > 1 and mean else float("nan")
        print("{:>10.3f}  {:>4d}  {:>10.3f}  {:>9.3f}  {:>8.1f}%  {:>12.3f}"
              .format(vx, len(vals), mean, sd,
                      rel * 100 if rel == rel else float("nan"), mean / vx))
        xs.append(vx)
        ys.append(mean)
        for v in vals:
            rel_devs.append(abs(v - mean) / mean if mean else 0.0)
        if len(vals) < args.min_runs:
            thin.append(vx)

    for vx in thin:
        print("\nWARN  only {} run(s) at vx={:.2f}. The spread from one or two runs is "
              "not a spread.".format(len(by_speed[vx]), vx))

    if len(xs) < 2:
        print("\nOnly one commanded speed measured, so the proportionality that "
              "mission_ctl assumes cannot be tested. v_cal below is that single "
              "point scaled, which is exactly the assumption in question.")
        v_cal = ys[0] * (args.cruise / xs[0])
        slope = ys[0] / xs[0]
    else:
        slope = fit_through_origin(xs, ys)
        a, b = fit_free(xs, ys)
        r_prop = rss(xs, ys, lambda x: slope * x)
        r_free = rss(xs, ys, lambda x: a * x + b)
        print("\n=== is achieved speed proportional to commanded? ===")
        print("  through origin  achieved = {:.4f} x commanded        residual {:.3e}"
              .format(slope, r_prop))
        print("  free intercept  achieved = {:.4f} x commanded {:+.4f}  residual {:.3e}"
              .format(a, b, r_free))
        # mission_ctl multiplies v_cal by (vx / cruise_vx). If the free fit is much
        # better, that scaling is wrong away from the calibration point, and the
        # error is worst at the extremes of the trained envelope.
        if r_prop > 4.0 * max(r_free, 1e-12):
            lo, hi = min(xs), max(xs)
            worst = max(abs((slope * x) - (a * x + b)) for x in (lo, hi))
            print("\n  WARN  the free-intercept fit is much better, so achieved speed is"
                  "\n        NOT proportional to commanded. mission_ctl scales v_cal by"
                  "\n        vx / cruise_vx, and over {:.2f}..{:.2f} m/s that model is off"
                  "\n        by up to {:.3f} m/s ({:.0f}%). Either keep commands near"
                  "\n        cruise_vx, or calibrate per speed."
                  .format(lo, hi, worst, 100 * worst / max(slope * hi, 1e-9)))
        else:
            print("\n  the proportional model is consistent with these readings, so"
                  "\n  mission_ctl's v_cal x (vx / cruise_vx) is justified here.")
        v_cal = slope * args.cruise
        if not (min(xs) <= args.cruise <= max(xs)):
            print("\n  NOTE  cruise {:.2f} m/s is OUTSIDE the measured range "
                  "{:.2f}..{:.2f}; v_cal below is extrapolated."
                  .format(args.cruise, min(xs), max(xs)))

    # The error bar: the largest relative deviation seen, floored so a lucky pair
    # of runs cannot produce an implausibly tight claim.
    observed = max(rel_devs) if rel_devs else 0.0
    rel_err = max(0.05, round(observed, 3))
    print("\n=== mission.yaml ===")
    print("  cruise_vx: {:.2f}".format(args.cruise))
    print("  v_cal: {:.3f}          # measured, tape + stopwatch, {} runs".format(
        v_cal, len(rows)))
    print("  v_cal_rel_err: {:.3f}  # worst observed relative deviation {:.1%}{}"
          .format(rel_err, observed,
                  ", floored at 5%" if rel_err > observed else ""))
    print("\n  -> \"walk 10m\" becomes {:.1f} s and is reported as 10.0 +- {:.1f} m"
          .format(10.0 / v_cal if v_cal > 0 else float("nan"), 10.0 * rel_err))
    return 0


if __name__ == "__main__":
    sys.exit(main())
