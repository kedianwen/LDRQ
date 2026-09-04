#!/usr/bin/env python3
"""Read the bridge's /r1_hw_bridge/joint_pos with the joint NAMES attached.

The bridge publishes a bare 26-float array. Counting to index 11 by eye to check
whether the left knee moved is precisely the transcription error the joint map
was built to avoid, so don't: this prints the name beside every value and calls
out what changed.

  python3 watch_obs.py                     # names beside values, call out what moved
  python3 watch_obs.py --check-default     # stage D3: prove the default pose is subtracted
"""
import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent


def load_names():
    """Names come from joints.tsv, which gen_joints.py derives from
    policy_interface.json. Never hand-typed here."""
    names = []
    for line in (HERE / "joints.tsv").read_text().splitlines():
        if line and not line.startswith("#"):
            _, name, _ = line.split("\t")
            names.append(name)
    return names


def load_defaults():
    """Defaults come from the same generated interface spec the bridge header is
    built from -- never hand-typed here."""
    spec = json.loads((HERE.parents[1] / "interface" / "policy_interface.json").read_text())
    return spec["articulation"]["default_joint_pos"]


def check_default(args, names):
    """Stage D3, without assuming anything about the robot's pose.

    While the policy is not driving, the bridge commands every joint to hold
    where it already is, so ~/cmd_debug carries the RAW absolute angle while
    ~/joint_pos carries the same angle minus the default pose. Their difference
    must therefore be the default pose itself -- a 26-way exact check that needs
    no knowledge of how the limbs happen to be hanging.

    The pair has to be sampled at the same instant: in damping mode a limp arm
    drifts, and two captures a minute apart differ by more than the tolerance.
    """
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import Float32MultiArray

    defaults = load_defaults()
    if len(defaults) != len(names):
        print(f"[fail] interface has {len(defaults)} defaults, joints.tsv has {len(names)}")
        return 1

    class Check(Node):
        def __init__(self):
            super().__init__("r1_check_default")
            self.jp = self.cd = None
            self.jp_t = self.cd_t = 0.0
            self.done = False
            self.create_subscription(Float32MultiArray, args.topic,
                                     self.on_jp, qos_profile_sensor_data)
            self.create_subscription(Float32MultiArray, args.cmd_topic,
                                     self.on_cd, qos_profile_sensor_data)
            print(f"pairing {args.topic} with {args.cmd_topic} "
                  f"(within {args.pair_ms:.0f} ms)...")
            print("NOTE: only valid while the policy is NOT driving -- run this in")
            print("      stage C/D, with enable_output=false and no policy node.\n")
            self.create_timer(5.0, self.timeout)

        def on_jp(self, m):
            self.jp, self.jp_t = list(m.data), time.monotonic()
            self.try_pair()

        def on_cd(self, m):
            self.cd, self.cd_t = list(m.data), time.monotonic()
            self.try_pair()

        def timeout(self):
            if not self.done:
                print(f"[fail] no simultaneous pair in 5 s. jp={'yes' if self.jp else 'NO'} "
                      f"cmd_debug={'yes' if self.cd else 'NO'}")
                print("       cmd_debug is published at ~10 Hz; if it never arrives the")
                print("       bridge is not reaching SendCommand (no LowState?).")
                self.done = True
                rclpy.shutdown()

        def try_pair(self):
            if self.done or self.jp is None or self.cd is None:
                return
            if abs(self.jp_t - self.cd_t) * 1000.0 > args.pair_ms:
                return
            self.done = True
            self.report()
            rclpy.shutdown()

        def report(self):
            if len(self.jp) != len(names) or len(self.cd) != len(names):
                print(f"[fail] sizes: joint_pos={len(self.jp)} cmd_debug={len(self.cd)} "
                      f"expected {len(names)}")
                return
            bad = []
            print(f"  {'#':>2} {'joint':<30} {'raw q':>9} {'obs':>9} "
                  f"{'diff':>9} {'default':>9}")
            for i, n in enumerate(names):
                diff = self.cd[i] - self.jp[i]
                ok = abs(diff - defaults[i]) <= args.tol
                if not ok:
                    bad.append((i, n, diff, defaults[i]))
                print(f"  {i:>2} {n:<30} {self.cd[i]:>+9.4f} {self.jp[i]:>+9.4f} "
                      f"{diff:>+9.4f} {defaults[i]:>+9.4f} {'' if ok else '  <-- MISMATCH'}")
            print()
            if bad:
                print(f"[FAIL] {len(bad)}/{len(names)} joints: cmd_debug - joint_pos "
                      f"!= default (tol {args.tol})")
                print("       The bridge is not subtracting the default pose the way the")
                print("       policy was trained. Do NOT go on to stage E.")
            else:
                print(f"[ok] 26/26: cmd_debug - joint_pos == default pose "
                      f"(within {args.tol} rad). Default subtraction confirmed.")

    rclpy.init()
    node = Check()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/r1_hw_bridge/joint_pos")
    ap.add_argument("--thresh", type=float, default=0.02,
                    help="rad of change before a joint is called out")
    ap.add_argument("--check-default", action="store_true",
                    help="stage D3: verify the default pose is subtracted from joint_pos")
    ap.add_argument("--cmd-topic", default="/r1_hw_bridge/cmd_debug")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="rad tolerance for --check-default")
    ap.add_argument("--pair-ms", type=float, default=30.0,
                    help="max age difference between the two samples")
    args = ap.parse_args()

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from std_msgs.msg import Float32MultiArray
    except ImportError as e:
        print(f"[fail] {e}\n  source /opt/ros/$ROS_DISTRO/setup.bash first")
        return 1

    names = load_names()

    if args.check_default:
        return check_default(args, names)

    class Watch(Node):
        def __init__(self):
            super().__init__("r1_watch_obs")
            self.base = None
            self.got = 0
            self.create_subscription(Float32MultiArray, args.topic,
                                     self.cb, qos_profile_sensor_data)
            print(f"listening on {args.topic} ({len(names)} joints)")
            # Sitting silently is the failure mode this project keeps hitting.
            # count_publishers separates the two causes outright: zero means
            # discovery never saw the bridge, non-zero means it did and the
            # messages are being dropped on QoS.
            self.create_timer(3.0, self.diagnose)

        def diagnose(self):
            if self.got:
                return
            import os
            n = self.count_publishers(args.topic)
            print(f"\n[!] 3 秒内没收到消息。{args.topic} 上可见的发布者: {n}")
            print(f"    本终端: ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '(unset->0)')}"
                  f"  ROS_LOCALHOST_ONLY={os.environ.get('ROS_LOCALHOST_ONLY', '(unset->0)')}"
                  f"  RMW={os.environ.get('RMW_IMPLEMENTATION', '(default)')}")
            if n == 0:
                print("    -> 发现阶段就没看到 bridge。两个终端的 ROS_DOMAIN_ID /")
                print("       ROS_LOCALHOST_ONLY 必须一致，且都要 `source env.sh`。")
                print("       对照 bridge 启动日志的第三行 `ROS side: ...`。")
            else:
                print("    -> 发布者可见但消息没到，是 QoS 不匹配。本脚本用的是")
                print("       sensor_data(best_effort)，与 bridge 一致，不该发生 —— ")
                print("       检查 bridge 是不是停在 WAITING_LOWSTATE（没收到机器人数据）。")

        def cb(self, msg):
            self.got += 1
            v = list(msg.data)
            if len(v) != len(names):
                print(f"[fail] got {len(v)} values, joints.tsv has {len(names)}"
                      " -- the bridge and the interface spec disagree")
                rclpy.shutdown()
                return
            if self.base is None:
                self.base = v
                print("baseline recorded. move one joint.\n")
                for i, (n, x) in enumerate(zip(names, v)):
                    print(f"  {i:>2} {n:<30} {x:+.4f}")
                print()
                return
            moved = [(abs(a - b), i) for i, (a, b) in enumerate(zip(v, self.base))]
            moved.sort(reverse=True)
            if moved[0][0] > args.thresh:
                cells = " | ".join(
                    f"{i} {names[i]} {v[i] - self.base[i]:+.3f}"
                    for d, i in moved[:3] if d > args.thresh)
                print(f"\r{cells:<110}", end="", flush=True)

    rclpy.init()
    node = Watch()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
