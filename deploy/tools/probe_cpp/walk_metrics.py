#!/usr/bin/env python3
"""First real-robot metrics: survival, tracking error, torque headroom, smoothness.

This is the PG-2 instrument. It answers four questions that the W08 gate is
phrased in terms of, and it is careful about one it CANNOT answer.

  survival     how long the stack stayed RUNNING, and how far the torso tilted
  tracking     commanded joint angle minus measured, per joint and per group
  torque       measured tau_est against each joint's rating -- headroom, not guesswork
  smoothness   |a_t - a_{t-1}| and a chatter rate, which is the evidence the
               "do we need an action low-pass filter?" question has been waiting
               for. A gait with no chatter has no measured reason for a filter.

What it cannot answer: ACHIEVED WALKING SPEED. The 425-dim observation has no
base linear velocity term (base_ang_vel, projected_gravity, velocity_commands,
joint_pos, joint_vel, actions) and the robot has no odometry publisher, so there
is nothing in the graph to integrate. Speed comes from a tape measure and a
stopwatch; pass --distance-m and --walk-seconds and this tool will do the
arithmetic and compare it against what was COMMANDED. Without them it reports
the commanded range only and says so, rather than inventing a number.

  # on the robot, while walking, developer mode, spotter on the e-stop
  python3 walk_metrics.py --seconds 60 --save ~/orin_commissioning/walk_pg2.json \
      --note "gantry slack, 0.3 m/s forward"

  # then, with the tape measure reading
  python3 walk_metrics.py --load ~/orin_commissioning/walk_pg2.json \
      --distance-m 4.2 --walk-seconds 15.0
"""
import argparse
import bisect
import json
import math
import pathlib
import re
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
DEPLOY = HERE.parents[1]
JOINT_MAP = DEPLOY / "ros2_ws" / "src" / "r1_hw_bridge" / "include" / "r1_hw_bridge" / "joint_map.hpp"
SPEC = DEPLOY / "interface" / "policy_interface.json"

# Torso-upright projected gravity, measured 2026-09-02. Tilt is reported against
# THIS, not against (0,0,-1): the IMU is not mounted perfectly level, and using
# the ideal vector would report ~3.5 deg of tilt on a robot standing straight.
UPRIGHT_GRAVITY = (0.060, 0.016, -0.998)


def parse_hpp_array(name):
    """Pull one `std::array<float, kNumJoints> <name> = { v, // i joint [group] }`
    out of joint_map.hpp.

    Parsed rather than copied. The gains are already generated from the actuator
    dump into that header; a second hand-maintained copy in Python would be one
    more thing that can silently go stale, and this project has already paid for
    exactly that failure once (the per-joint gain export).
    """
    text = JOINT_MAP.read_text()
    m = re.search(rf"{name}\s*=\s*\{{(.*?)\}};", text, re.S)
    if not m:
        sys.exit(f"{JOINT_MAP.name}: could not find {name}")
    vals, names, groups = [], [], []
    for line in m.group(1).splitlines():
        vm = re.match(r"\s*(-?[\d.]+)f\s*,\s*//\s*\d+\s+(\S+)\s*(?:\[(\w+)\])?", line)
        if vm:
            vals.append(float(vm.group(1)))
            names.append(vm.group(2))
            groups.append(vm.group(3) or "-")
    if not vals:
        sys.exit(f"{JOINT_MAP.name}: {name} parsed to nothing -- has its format changed?")
    return vals, names, groups


def load_spec():
    spec = json.loads(SPEC.read_text())
    obs = spec["observation"]
    hist = obs["history_length"]
    terms, off = {}, 0
    for t in obs["terms"]:
        w = t["width"]
        terms[t["name"]] = {"width": w, "newest": off + (hist - 1) * w}
        off += w * hist
    # Action index != articulation index: head_pitch and head_yaw are not
    # policy-actuated, so action 8 is articulation 9. Labelling the action table
    # with the 26-entry joint list would silently mis-name every action from 8 up.
    return spec["articulation"]["default_joint_pos"], terms, spec["action"]["joint_names"]


def newest(sample, term):
    return sample[term["newest"]:term["newest"] + term["width"]]


def nearest(times, series, t):
    """Value in `series` whose timestamp is closest to t.

    cmd_debug arrives at ~10 Hz and joint_pos at 50 Hz, so zipping them by index
    would pair a command with a measurement up to seconds away once the streams
    drift. Pairing by time keeps each tracking-error sample meaningful on its own
    instead of only in aggregate.
    """
    if not times:
        return None
    i = bisect.bisect_left(times, t)
    if i == 0:
        best = 0
    elif i >= len(times):
        best = len(times) - 1
    else:
        best = i if (times[i] - t) < (t - times[i - 1]) else i - 1
    return series[best]


def pstat(vals):
    if not vals:
        return None
    s = sorted(vals)
    return {
        "n": len(s), "mean": statistics.fmean(s), "p50": s[len(s) // 2],
        "p95": s[min(len(s) - 1, int(0.95 * len(s)))], "max": s[-1],
    }


def fmt(st, unit="", w=9):
    if st is None:
        return "(no data)"
    return (f"mean {st['mean']:{w}.4f}{unit}  p50 {st['p50']:{w}.4f}{unit}"
            f"  p95 {st['p95']:{w}.4f}{unit}  max {st['max']:{w}.4f}{unit}  n={st['n']}")


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def section_survival(blob):
    print("\n=== A. survival ===")
    status = blob["status"]
    t0, t1 = blob["t_start"], blob["t_end"]
    print(f"  recording window            {t1 - t0:.1f} s")
    if not status:
        print("  no ~/status messages. The topic is transient_local+reliable, so an")
        print("  empty read means the bridge is not up -- not that it is healthy.")
    # Longest run of consecutive RUNNING, bounded by the recording itself: a
    # window that ends while still RUNNING gives a LOWER BOUND on survival, and
    # that distinction is the whole PG-2 claim.
    best, cur_start, ended_running = 0.0, None, False
    first_degraded = None
    for t, text in status:
        running = text.startswith("RUNNING")
        if running and cur_start is None:
            cur_start = t
        elif not running and cur_start is not None:
            best = max(best, t - cur_start)
            cur_start = None
            if first_degraded is None and text.startswith("DEGRADED"):
                first_degraded = (t - t0, text)
    if cur_start is not None:
        best = max(best, t1 - cur_start)
        ended_running = True
    print(f"  longest continuous RUNNING  {best:.1f} s"
          f"{'  (still RUNNING at the end -- this is a LOWER BOUND)' if ended_running else ''}")
    if first_degraded:
        print(f"  first DEGRADED at +{first_degraded[0]:.1f} s: {first_degraded[1]}")

    grav = blob["imu"]
    if grav:
        n = math.sqrt(sum(c * c for c in UPRIGHT_GRAVITY))
        tilts = []
        for _, v in grav:
            g = v[7:10]
            m = math.sqrt(sum(c * c for c in g))
            if m < 1e-6:
                continue
            dot = sum(a * b for a, b in zip(g, UPRIGHT_GRAVITY)) / (m * n)
            tilts.append(math.degrees(math.acos(max(-1.0, min(1.0, dot)))))
        if tilts:
            over20 = sum(1 for x in tilts if x > 20.0) / len(tilts)
            print(f"  torso tilt vs upright       {fmt(pstat(tilts), ' deg')}")
            print(f"  fraction of time past 20 deg {100 * over20:.1f}%")
    print("\n  PG-2 is 60 s of continuous flat walking. Two things this recording")
    print("  CANNOT establish, so they go in the log by hand:")
    print("    - whether a spotter took any weight (invisible here; the stack stays")
    print("      RUNNING through a catch)")
    print("    - whether the gantry was slack. 'RUNNING for 60 s' on a supported")
    print("      robot is not PG-2.")
    if blob.get("note"):
        print(f"  operator note: {blob['note']}")
    else:
        print("  ! no --note recorded. Add one; the numbers below are not interpretable")
        print("    without knowing the gantry state and the commanded speed.")


def section_speed(blob, terms, distance_m, walk_seconds):
    print("\n=== B. speed ===")
    obs = blob["obs"]
    if obs:
        vc = [newest(v, terms["velocity_commands"]) for _, v in obs]
        for i, axis in enumerate(("vx", "vy", "wz")):
            col = [s[i] for s in vc]
            print(f"  commanded {axis:<3}  min {min(col):+.3f}  max {max(col):+.3f}"
                  f"  mean {statistics.fmean(col):+.3f}")
    else:
        print("  no ~/obs samples, so not even the commanded velocity is known.")

    if distance_m is not None and walk_seconds:
        achieved = distance_m / walk_seconds
        print(f"\n  measured        {distance_m:.2f} m in {walk_seconds:.1f} s"
              f"  =>  {achieved:.3f} m/s")
        if obs:
            vx = [s[0] for s in vc]
            cmd_mean = statistics.fmean(vx)
            if abs(cmd_mean) > 0.02:
                print(f"  commanded mean vx {cmd_mean:+.3f} m/s"
                      f"  =>  achieved/commanded = {achieved / abs(cmd_mean):.2f}")
                print("  A ratio well under 1 is a velocity tracking gap: the policy is")
                print("  being asked for a speed the robot does not reach. That is a")
                print("  training/DR input, not something to fix with gains.")
    else:
        print("\n  ACHIEVED SPEED NOT MEASURED. There is no base linear velocity in the")
        print("  425-dim observation and no odometry topic, so it cannot be recovered")
        print("  from this recording at all -- not by integration, not by any")
        print("  post-processing. Mark a start and end line, time the walk, and re-run")
        print("  with --load ... --distance-m X --walk-seconds Y.")


def section_tracking(blob, defaults, kp, names, groups):
    print("\n=== C. joint tracking (commanded - measured, rad) ===")
    cmd, jpos = blob["cmd"], blob["jpos"]
    if not cmd or not jpos:
        print("  needs both ~/cmd_debug and ~/joint_pos.")
        return None
    jt = [t for t, _ in jpos]
    jv = [v for _, v in jpos]
    errs = [[] for _ in names]
    for t, c in cmd:
        m = nearest(jt, jv, t)
        if m is None:
            continue
        for j in range(len(names)):
            errs[j].append(c[j] - (m[j] + defaults[j]))

    by_group = {}
    for j, g in enumerate(groups):
        by_group.setdefault(g, []).extend(abs(e) for e in errs[j])
    for g in sorted(by_group):
        print(f"  [{g:<7}] |err|  {fmt(pstat(by_group[g]), ' rad', 8)}")

    print("\n  worst joints by mean |err|:")
    ranked = sorted(range(len(names)),
                    key=lambda j: statistics.fmean([abs(e) for e in errs[j]]) if errs[j] else 0.0,
                    reverse=True)
    for j in ranked[:5]:
        e = errs[j]
        # Signed mean alongside |mean|: a symmetric swing about zero and a
        # constant offset have the same |err| and completely different causes.
        # W07's narrow stance was the second kind -- a 0.129 rad DC offset.
        print(f"    {names[j]:<30} mean {statistics.fmean(e):+.4f}"
              f"  |mean| {statistics.fmean([abs(x) for x in e]):.4f}"
              f"  p2p {max(e) - min(e):.4f}")
    return errs


def section_torque(blob, errs, kp, tau_limit, names):
    print("\n=== D. torque headroom ===")
    tau = blob.get("tau") or []
    measured = False
    if tau:
        peak = max(max(abs(x) for x in v) for _, v in tau)
        if peak > 1e-6:
            measured = True
        else:
            print("  ~/joint_tau is present but identically zero: this firmware does not")
            print("  populate LowState.tau_est. Falling back to the kKp*error estimate,")
            print("  which is an ESTIMATE and ignores the kKd term and gravity.")
    else:
        print("  no ~/joint_tau samples (old bridge build, or the topic is not up).")
        print("  Falling back to the kKp*error ESTIMATE.")

    if measured:
        per = [[] for _ in names]
        for _, v in tau:
            for j in range(len(names)):
                per[j].append(abs(v[j]))
        label = "measured tau_est"
    elif errs:
        per = [[abs(kp[j] * e) for e in errs[j]] for j in range(len(names))]
        label = "kKp*err ESTIMATE"
    else:
        print("  nothing to work from.")
        return

    print(f"  source: {label}")
    rows = []
    for j in range(len(names)):
        if not per[j]:
            continue
        pk = max(per[j])
        frac = pk / tau_limit[j] if tau_limit[j] > 0 else 0.0
        over = sum(1 for x in per[j] if x > tau_limit[j]) / len(per[j])
        rows.append((frac, j, pk, over))
    rows.sort(reverse=True)
    print(f"  {'joint':<30}{'peak N*m':>10}{'rating':>9}{'% of rating':>13}{'frac over':>11}")
    for frac, j, pk, over in rows[:6]:
        print(f"  {names[j]:<30}{pk:10.2f}{tau_limit[j]:9.1f}{100 * frac:12.1f}%{100 * over:10.2f}%")
    worst = rows[0] if rows else None
    if worst:
        if worst[0] < 0.5:
            print(f"\n  Worst joint peaks at {100 * worst[0]:.0f}% of its rating: not saturating,")
            print("  and not close. A tracking error here is a load or a gain question,")
            print("  not a limit question.")
        elif worst[0] < 1.0:
            print(f"\n  Worst joint peaks at {100 * worst[0]:.0f}% of rating -- inside the limit but")
            print("  without much room. Worth watching before raising any kp.")
        else:
            print(f"\n  Worst joint EXCEEDS its rating ({100 * worst[0]:.0f}%). Stop; this is the")
            print("  one W08 acceptance criterion that is a hard fail.")


def section_smoothness(blob, action_names):
    print("\n=== E. action smoothness ===")
    act = blob["action"]
    if len(act) < 3:
        print("  needs ~/action from the policy node.")
        return
    times = [t for t, _ in act]
    vals = [v for _, v in act]
    dim = len(vals[0])
    span = times[-1] - times[0]
    rate = (len(act) - 1) / span if span > 0 else 0.0
    print(f"  {len(act)} actions over {span:.1f} s ({rate:.1f} Hz)")
    if rate > 70:
        print("  ! over 70 Hz on a 50 Hz loop means TWO policy nodes are publishing.")
        print("    Fix that before reading anything below (pkill -f, both instances).")

    step = [max(abs(vals[i][j] - vals[i - 1][j]) for j in range(dim))
            for i in range(1, len(vals))]
    print(f"  per-step |da|_inf   {fmt(pstat(step), '', 8)}")

    # Chatter, as a frequency rather than an impression.
    #
    # Count sign changes in the first difference. One full oscillation of the
    # action contributes two of them, so sign_changes_per_second / 2 estimates
    # the dominant frequency of whatever the action is doing. Checked against the
    # synthetic case: a 1.2 Hz sinusoid gives 2.4 flips/s -> 1.2 Hz.
    #
    # The ceiling is 50 flips/s, not 25: at 50 Hz a value that alternates every
    # sample makes the DIFFERENCE alternate every sample too, which is 25 Hz --
    # Nyquist. Reporting the implied frequency instead of the raw rate is what
    # makes the number comparable to the gait (1-2 Hz) and to the frequency a
    # low-pass filter would be set to.
    flips = []
    for j in range(dim):
        d = [vals[i][j] - vals[i - 1][j] for i in range(1, len(vals))]
        sc = sum(1 for i in range(1, len(d)) if d[i] * d[i - 1] < 0)
        flips.append(sc / span if span > 0 else 0.0)
    worst = max(range(dim), key=lambda j: flips[j])
    print(f"  sign-flip rate      mean {statistics.fmean(flips):.1f}/s"
          f"   max {flips[worst]:.1f}/s on action[{worst}]"
          f" ({action_names[worst] if worst < len(action_names) else '?'})")
    print(f"  implied dominant f  mean {statistics.fmean(flips) / 2:.1f} Hz"
          f"   max {flips[worst] / 2:.1f} Hz"
          f"   (gait is 1-2 Hz; Nyquist at 50 Hz is 25 Hz)")

    if flips[worst] < 8.0:
        print("\n  VERDICT INPUT: no chatter. Even the worst action's implied frequency")
        print("  sits in the gait band, so there is no measured basis for enabling the")
        print("  action low-pass filter. Write that down as 'measured, not needed' --")
        print("  it is a different deliverable from 'not done'.")
    elif flips[worst] < 20.0:
        print("\n  VERDICT INPUT: 4-10 Hz content, above the gait band. Before adding a")
        print("  filter,")
        print("  confirm developer mode and check setpoint lag -- both have produced")
        print("  this signature before, and a filter would mask them rather than fix")
        print("  them.")
    else:
        print("\n  VERDICT INPUT: chatter. FIRST confirm developer mode (a second writer")
        print("  on rt/lowcmd looks exactly like this -- W06 lost a session to it).")
        print("  Only if developer mode is confirmed does a low-pass have a basis.")


# ---------------------------------------------------------------------------

def collect(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, qos_profile_sensor_data, DurabilityPolicy, ReliabilityPolicy
    from std_msgs.msg import Float32MultiArray, String

    class Collector(Node):
        def __init__(self):
            super().__init__("r1_walk_metrics")
            self.d = {k: [] for k in ("jpos", "jvel", "tau", "cmd", "action", "imu", "obs", "status")}
            t = lambda: self.get_clock().now().nanoseconds * 1e-9
            def arr(key, topic):
                self.create_subscription(
                    Float32MultiArray, topic,
                    lambda m, k=key: self.d[k].append((t(), list(m.data))),
                    qos_profile_sensor_data)
            arr("jpos", args.prefix + "/joint_pos")
            arr("jvel", args.prefix + "/joint_vel")
            arr("tau", args.prefix + "/joint_tau")
            arr("cmd", args.prefix + "/cmd_debug")
            arr("imu", args.prefix + "/imu")
            arr("obs", args.prefix + "/obs")
            arr("action", args.policy_prefix + "/action")
            # ~/status is reliable + transient_local; subscribing with the
            # sensor-data profile would silently match nothing.
            self.create_subscription(
                String, args.prefix + "/status",
                lambda m: self.d["status"].append((t(), m.data)),
                QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL))

    rclpy.init()
    node = Collector()
    print(f"collecting {args.seconds:.0f}s ...")
    print("WALK THE ROBOT. Developer mode, spotter on the e-stop.")
    t0 = node.get_clock().now().nanoseconds * 1e-9
    end = t0 + args.seconds
    while rclpy.ok() and node.get_clock().now().nanoseconds * 1e-9 < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    t1 = node.get_clock().now().nanoseconds * 1e-9
    d = node.d
    node.destroy_node()
    rclpy.shutdown()

    for k, v in d.items():
        print(f"  {k:<8} {len(v)} samples")
    if not d["jpos"]:
        print("\nno joint_pos. Before suspecting the bridge:")
        print("  echo $ROS_LOCALHOST_ONLY   # must be 1 here AND in the stack's terminal")
        print("  ros2 daemon stop           # foxy caches a stale graph snapshot")
        print("  pgrep -f r1_hw_bridge      # -f is required; comm truncates to 15 chars")
        sys.exit(1)
    blob = dict(d)
    blob["t_start"], blob["t_end"] = t0, t1
    blob["note"] = args.note
    return blob


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--prefix", default="/r1_hw_bridge")
    ap.add_argument("--policy-prefix", default="/r1_policy_runner")
    ap.add_argument("--save")
    ap.add_argument("--load")
    ap.add_argument("--note", default="", help="gantry state and commanded speed -- "
                                              "the numbers are not interpretable without it")
    ap.add_argument("--distance-m", type=float, help="tape-measured distance walked")
    ap.add_argument("--walk-seconds", type=float, help="stopwatch time over that distance")
    args = ap.parse_args()

    defaults, terms, action_names = load_spec()
    kp, names, groups = parse_hpp_array("kKp")
    tau_limit, _, _ = parse_hpp_array("kTauLimit")

    if args.load:
        blob = json.loads(pathlib.Path(args.load).read_text())
        if args.note:
            blob["note"] = args.note
    else:
        blob = collect(args)
        if args.save:
            pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(args.save).write_text(json.dumps(blob))
            print(f"raw samples -> {args.save}")

    section_survival(blob)
    section_speed(blob, terms, args.distance_m, args.walk_seconds)
    errs = section_tracking(blob, defaults, kp, names, groups)
    section_torque(blob, errs, kp, tau_limit, names)
    section_smoothness(blob, action_names)
    print()


if __name__ == "__main__":
    main()
