#!/usr/bin/env python3
"""Identify which of the 35 `unitree_hg` motor slots this robot actually uses.

The SDK installed on the R1 ships joint-index enums for `go2` and `g1` only.
R1 is a 26-DoF humanoid and G1 is 29-DoF, so the G1 enum does NOT describe this
robot, and there is no `r1/defines.h` to read. The mapping therefore has to come
from the robot itself.

READ-ONLY. This subscribes to LowState and never publishes anything -- it cannot
command a joint. Safe to run against a powered robot with the factory stack up.

Two modes:

  survey (default)  Collect samples and report, per slot, whether it looks alive
                    (non-zero readings, plausible temperature, movement).
  --watch           Stream the slots that are moving right now. Move ONE joint by
                    hand and read off its index. This is the definitive test:
                    aliveness tells you which slots exist, moving a joint tells
                    you which slot IS that joint.

  --discover        Print what the installed SDK actually exposes and exit. Run
                    this first if the imports below fail.

Usage:
  python3 probe_lowstate.py --discover
  python3 probe_lowstate.py --iface eth10 --seconds 5
  python3 probe_lowstate.py --iface eth10 --watch
"""

import argparse
import math
import sys
import time

N_SLOTS = 35


def discover():
    """Report the shape of the installed SDK rather than guessing at it."""
    import importlib
    import pkgutil

    try:
        import unitree_sdk2py
    except ImportError as e:
        print(f"[fail] cannot import unitree_sdk2py: {e}")
        print("       try the SDK's own venv:")
        print("         source ~/unitree_sdk2_python/.venv/bin/activate")
        return 1

    print(f"unitree_sdk2py at {unitree_sdk2py.__file__}")
    for modname in ("unitree_sdk2py.idl.unitree_hg.msg.dds_",
                    "unitree_sdk2py.idl.unitree_go.msg.dds_"):
        try:
            m = importlib.import_module(modname)
            names = [n for n in dir(m) if not n.startswith("_")]
            print(f"  {modname}: {', '.join(names[:12])}")
        except ImportError as e:
            print(f"  {modname}: NOT AVAILABLE ({e})")

    try:
        idl = importlib.import_module("unitree_sdk2py.idl")
        print("  idl subpackages:",
              ", ".join(n for _, n, _ in pkgutil.iter_modules(idl.__path__)))
    except Exception as e:  # noqa: BLE001 - diagnostic path, report anything
        print(f"  (could not enumerate idl subpackages: {e})")
    return 0


def quat_rotate_inverse(q, v):
    """Isaac Lab's quat_rotate_inverse, for q = (w, x, y, z)."""
    w, x, y, z = q
    vec = (x, y, z)
    dot_qv = sum(a * b for a, b in zip(vec, v))
    cross = (vec[1] * v[2] - vec[2] * v[1],
             vec[2] * v[0] - vec[0] * v[2],
             vec[0] * v[1] - vec[1] * v[0])
    return tuple(v[i] * (2.0 * w * w - 1.0) - cross[i] * w * 2.0 + vec[i] * dot_qv * 2.0
                 for i in range(3))


def report_imu(imu):
    """The bridge has to feed the policy base angular velocity and projected
    gravity, so the IMU convention matters as much as the joint order."""
    q = list(imu.quaternion)
    g = list(imu.gyroscope)
    a = list(imu.accelerometer)
    print("\n=== IMU ===")
    print(f"  quaternion   {['%+.4f' % x for x in q]}")
    print(f"  gyroscope    {['%+.4f' % x for x in g]}  <- policy obs `base_ang_vel`")
    print(f"  accelerometer{['%+.4f' % x for x in a]}")
    if hasattr(imu, "rpy"):
        print(f"  rpy          {['%+.4f' % x for x in list(imu.rpy)]}")

    # Which quaternion ordering is it? Standing upright, projected gravity in the
    # base frame must be close to (0, 0, -1). Only one ordering gives that.
    down = (0.0, 0.0, -1.0)
    wxyz = quat_rotate_inverse((q[0], q[1], q[2], q[3]), down)
    xyzw = quat_rotate_inverse((q[3], q[0], q[1], q[2]), down)
    print("\n  projected gravity, assuming the array is:")
    print(f"    (w,x,y,z) -> {['%+.4f' % x for x in wxyz]}")
    print(f"    (x,y,z,w) -> {['%+.4f' % x for x in xyzw]}")
    print("  With the robot UPRIGHT the correct ordering reads about (0, 0, -1).")
    print("  That ordering is what the bridge must use. Do not guess it.")


def survey(samples):
    """Per-slot aliveness. A slot that never moves and reads exactly zero on
    every channel is almost certainly not a real joint on this robot."""
    print(f"\n=== 槽位普查（{len(samples)} 帧）===")
    print(f"{'idx':>4} {'q':>9} {'dq':>9} {'tau':>9} {'temp':>6} "
          f"{'mode':>5} {'q范围':>9}  判定")

    live = []
    for i in range(N_SLOTS):
        qs = [s[i].q for s in samples]
        dqs = [s[i].dq for s in samples]
        taus = [s[i].tau_est for s in samples]
        temp = samples[-1][i].temperature
        temp = temp[0] if isinstance(temp, (list, tuple)) else temp
        mode = samples[-1][i].mode
        span = max(qs) - min(qs)

        nonzero = any(abs(x) > 1e-9 for x in qs + dqs + taus)
        warm = isinstance(temp, (int, float)) and temp > 0
        alive = nonzero or warm or mode != 0
        if alive:
            live.append(i)

        print(f"{i:>4} {qs[-1]:>+9.4f} {dqs[-1]:>+9.4f} {taus[-1]:>+9.4f} "
              f"{temp:>6} {mode:>5} {span:>9.5f}  {'LIVE' if alive else '-'}")

    print(f"\n判定为 LIVE 的槽位（{len(live)} 个）: {live}")
    print("我们的策略是 24 维动作 / 26 维关节观测。")
    if len(live) == 26:
        print("  -> 26 个，与关节观测维度吻合。用 --watch 逐个确认语义。")
    elif len(live) == 24:
        print("  -> 24 个，与动作维度吻合（头部两关节可能不在 LowState 里）。")
    else:
        print(f"  -> {len(live)} 个，与 24/26 都对不上。别急着写映射，先把这个搞清楚。")
    print("\n温度为 0、mode 为 0、读数恒为零的槽位是 G1 布局里 R1 没用上的保留位。")


def watch(sub, read_one):
    print("\n=== 实时监视 ===")
    print("用手慢慢掰动**一个**关节，读出跳动的那个索引。Ctrl-C 结束。\n")
    base = None
    try:
        while True:
            msg = read_one(sub)
            if msg is None:
                continue
            ms = msg.motor_state
            if base is None:
                base = [ms[i].q for i in range(N_SLOTS)]
                print("已记录静止基准。现在开始掰。\n")
                continue
            moved = sorted(
                ((abs(ms[i].q - base[i]), i) for i in range(N_SLOTS)),
                reverse=True)[:5]
            if moved[0][0] > 0.01:
                cells = " | ".join(
                    f"idx {i:>2}: Δ{d:+.3f} rad, dq {ms[i].dq:+.3f}"
                    for d, i in moved if d > 0.01)
                print(f"\r{cells:<110}", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n结束。")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="eth10", help="机器人网口（阶段 2 实测为 eth10）")
    ap.add_argument("--domain", type=int, default=0, help="DDS domain id")
    ap.add_argument("--topic", default="rt/lowstate")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--watch", action="store_true", help="实时监视哪个索引在动")
    ap.add_argument("--discover", action="store_true", help="只打印 SDK 结构后退出")
    args = ap.parse_args()

    if args.discover:
        return discover()

    print("=" * 70)
    print("只读探针：仅订阅 LowState，不发布任何指令，无法驱动关节。")
    print("=" * 70)

    try:
        from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                                 ChannelSubscriber)
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    except ImportError as e:
        print(f"[fail] {e}\n先跑 --discover；必要时 source ~/unitree_sdk2_python/.venv/bin/activate")
        return 1

    ChannelFactoryInitialize(args.domain, args.iface)
    sub = ChannelSubscriber(args.topic, LowState_)
    sub.Init()

    def read_one(s):
        return s.Read(1000)

    first = None
    deadline = time.time() + 5.0
    while time.time() < deadline and first is None:
        first = read_one(sub)
    if first is None:
        print(f"[fail] 5 秒内没收到 {args.topic}。")
        print("  - 确认出厂运控在跑: ps aux | grep master_service")
        print(f"  - 网口对不对: ip -brief addr   (阶段 2 实测 {args.iface})")
        print("  - 换个话题名再试: --topic rt/lowstate_hg / lowstate")
        return 1

    print(f"\n已收到 {args.topic}。motor_state 槽位数: {len(first.motor_state)}")
    report_imu(first.imu_state)

    if args.watch:
        watch(sub, read_one)
        return 0

    samples, end = [], time.time() + args.seconds
    while time.time() < end:
        m = read_one(sub)
        if m is not None:
            samples.append(m.motor_state)
    if not samples:
        print("[fail] 首帧之后再没收到数据")
        return 1
    survey(samples)
    print("\n下一步: 加 --watch 逐个关节确认语义，把输出发回来写映射表。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
