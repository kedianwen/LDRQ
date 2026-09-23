#!/usr/bin/env python3
"""Is the narrow stance the policy's fault or the hardware's?

W06 closed with the robot walking, but with the feet too close together. That
has two completely different causes and they need opposite fixes:

  * the policy COMMANDS a narrow stance  -> a training problem (W08 retrain),
    nothing on the robot will fix it;
  * the policy commands a wide stance and the hardware DOESN'T GET THERE
    -> a deployment problem (gains, load, saturation), fixable this week.

Both look identical from outside the robot, so measure instead of guessing:
compare the commanded hip_roll against the measured hip_roll. If they agree,
the policy is asking for this stance.

  python3 analyze_stance.py --seconds 30

Run it while walking, in developer mode. Stance width is dominated by the hip
roll pair, so that is what this reports; hip_yaw is printed too because a
toe-in/toe-out asymmetry shows up there and not in roll.

Note ~/cmd_debug is decimated to ~cmd_rate/50 (about 10 Hz), which is plenty
for the mean offsets this tool reports but NOT enough to measure phase or lag.
Setpoint lag is reported by the bridge itself -- see its status line.
"""
import argparse
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent

# Stance geometry lives in these joints. Names, never indices -- indices are
# what the joint map exists to stop us hand-transcribing.
PAIRS = [
    ("left_hip_roll_joint", "right_hip_roll_joint"),
    ("left_hip_yaw_joint", "right_hip_yaw_joint"),
    ("left_hip_pitch_joint", "right_hip_pitch_joint"),
    ("left_ankle_roll_joint", "right_ankle_roll_joint"),
]


def load_names():
    names = []
    for line in (HERE / "joints.tsv").read_text().splitlines():
        if line and not line.startswith("#"):
            _, name, _ = line.split("\t")
            names.append(name)
    return names


def load_defaults():
    spec = json.loads((HERE.parents[1] / "interface" / "policy_interface.json").read_text())
    return spec["articulation"]["default_joint_pos"]


def summarise(name, samples, indent="  "):
    """Mean +- sd and peak-to-peak. Pure stdlib: numpy is not guaranteed on the
    robot and this tool has to run there."""
    if not samples:
        return f"{indent}{name:<34} (no data)"
    mean = statistics.fmean(samples)
    sd = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    return (f"{indent}{name:<34} {mean:+.4f} +- {sd:.4f} rad"
            f"   p2p {max(samples) - min(samples):.4f}   n={len(samples)}")


def report(jpos, cmd, names, defaults):
    """jpos holds OBSERVATION values (default already subtracted by the bridge);
    cmd holds ABSOLUTE targets. Add the default back before comparing, or every
    joint with a non-zero default reads as a large tracking error."""
    idx = {n: i for i, n in enumerate(names)}
    print("\n=== stance geometry (absolute rad) ===")
    print("  measured = joint_pos + default   commanded = cmd_debug (already absolute)\n")

    verdict = []
    for left, right in PAIRS:
        li, ri = idx[left], idx[right]
        m_sep = [(a[li] + defaults[li]) - (a[ri] + defaults[ri]) for a in jpos]
        print(f"{left.replace('left_', '')}:")
        print(summarise("measured  left-right separation", m_sep))
        if cmd:
            c_sep = [a[li] - a[ri] for a in cmd]
            print(summarise("commanded left-right separation", c_sep))
            if m_sep and c_sep:
                gap = statistics.fmean(c_sep) - statistics.fmean(m_sep)
                print(f"    commanded minus measured: {gap:+.4f} rad")
                verdict.append((left, statistics.fmean(c_sep), statistics.fmean(m_sep), gap))
        print()

    # Per-joint tracking error on the roll pair, the one that sets stance width.
    if cmd:
        n = min(len(jpos), len(cmd))
        print("=== hip_roll tracking (commanded vs measured, same joint) ===")
        for jn in ("left_hip_roll_joint", "right_hip_roll_joint"):
            j = idx[jn]
            err = [cmd[i][j] - (jpos[i][j] + defaults[j]) for i in range(n)]
            print(summarise(jn, err))
        print()

    if verdict:
        left, c, m, gap = verdict[0]
        print("=== reading ===")
        if abs(gap) < 0.02:
            print("  Commanded and measured stance agree (within 0.02 rad).")
            print("  => THE POLICY IS ASKING FOR THIS STANCE. Not a deployment")
            print("     problem; do not chase it with gains. It is a W08 retrain")
            print("     input (reward / command range / DR).")
        else:
            print(f"  Commanded stance differs from measured by {gap:+.4f} rad.")
            print("  => THE HARDWARE IS NOT REACHING THE COMMANDED STANCE.")
            print("     Deployment-side: check hip_roll kp against the load, and")
            print("     whether the effort limit is saturating. Fixable this week.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--topic", default="/r1_hw_bridge/joint_pos")
    ap.add_argument("--cmd-topic", default="/r1_hw_bridge/cmd_debug")
    ap.add_argument("--save", help="write raw samples here as JSON")
    ap.add_argument("--load", help="re-analyse a --save file instead of subscribing")
    args = ap.parse_args()

    names, defaults = load_names(), load_defaults()

    if args.load:
        blob = json.loads(pathlib.Path(args.load).read_text())
        report(blob["jpos"], blob["cmd"], names, defaults)
        return

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import Float32MultiArray

    class Collector(Node):
        def __init__(self):
            super().__init__("r1_stance_probe")
            self.jpos, self.cmd = [], []
            self.create_subscription(Float32MultiArray, args.topic,
                                     lambda m: self.jpos.append(list(m.data)),
                                     qos_profile_sensor_data)
            self.create_subscription(Float32MultiArray, args.cmd_topic,
                                     lambda m: self.cmd.append(list(m.data)),
                                     qos_profile_sensor_data)

    rclpy.init()
    node = Collector()
    print(f"collecting {args.seconds:.0f}s from {args.topic} and {args.cmd_topic}...")
    print("walk the robot during this window (developer mode, gantry attached).")
    end = node.get_clock().now().nanoseconds + int(args.seconds * 1e9)
    while rclpy.ok() and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    jpos, cmd = node.jpos, node.cmd
    node.destroy_node()
    rclpy.shutdown()

    print(f"got {len(jpos)} joint_pos and {len(cmd)} cmd_debug samples")
    if not jpos:
        # Same failure the W06 session lost an hour to. Say the cause here.
        print("\nno joint_pos samples. Before suspecting the bridge:")
        print("  ros2 daemon stop        # foxy caches a stale graph snapshot")
        print("  pgrep -f r1_hw_bridge   # -f is required; comm is truncated to 15 chars")
        sys.exit(1)
    if not cmd:
        print("\nno cmd_debug samples -- is the policy driving? Stance numbers")
        print("below are measured-only; the commanded/measured verdict needs both.")

    if args.save:
        pathlib.Path(args.save).write_text(json.dumps({"jpos": jpos, "cmd": cmd}))
        print(f"raw samples -> {args.save}")
    report(jpos, cmd, names, defaults)


if __name__ == "__main__":
    main()
