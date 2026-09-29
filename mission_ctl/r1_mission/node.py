"""The ROS shell and the CLI. Thin on purpose: all the logic is in plan.py,
yaw.py and executor.py, which are importable and tested without ROS.

ROS 2 foxy / Python 3.8 on the robot, std_msgs only -- no new message packages,
no geometry_msgs, nothing to build.

One invocation = one process = one plan. There is no daemon, and that is a safety
property rather than a simplification: the bridge decays a stale ~/cmd_vel to zero
after 500 ms, so a mission process that exits, crashes or is killed stops the
robot without having to remember to.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

from . import executor as ex
from . import plan as P
from . import yaw as Y

BRIDGE_NS = "/r1_hw_bridge"
CMD_TOPIC = BRIDGE_NS + "/cmd_vel"
IMU_TOPIC = BRIDGE_NS + "/imu"
STATUS_TOPIC = BRIDGE_NS + "/status"

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_CFG = HERE.parent / "config" / "mission.yaml"


def read_flat_yaml(path):
    """A flat `key: value` reader. Deliberately not PyYAML: the config has no
    nesting, and the robot's package situation is not one to add imports to."""
    out = {}
    for raw in pathlib.Path(str(path)).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if not v:
            continue
        try:
            out[k.strip()] = float(v) if ("." in v or "e" in v.lower()) else int(v)
        except ValueError:
            out[k.strip()] = v.strip('"\'')
    return out


def resolve_envelope(explicit=None):
    """Prefer the envelope the installed bundle declares, then the bridge's own
    config. Same numbers the bridge clamps to, read from the same place -- so the
    capability statement cannot drift out of step with what is enforced."""
    if explicit:
        return P.Envelope.from_bundle(explicit)
    root = os.environ.get("R1_DEPLOY_ROOT")
    if root:
        cand = pathlib.Path(root) / "interface" / "command_envelope.json"
        if cand.is_file():
            return P.Envelope.from_bundle(cand)
        cand = (pathlib.Path(root) / "ros2_ws/src/r1_hw_bridge/config/bridge.yaml")
        if cand.is_file():
            return P.Envelope.from_bridge_yaml(cand)
    repo = HERE.parents[1] / "deploy/ros2_ws/src/r1_hw_bridge/config/bridge.yaml"
    if repo.is_file():
        return P.Envelope.from_bridge_yaml(repo)
    raise P.PlanError(
        "cannot find a command envelope. Set R1_DEPLOY_ROOT (source env.sh) or "
        "pass --envelope <command_envelope.json>.")


UNMEASURED = ("", "none", "null", "unmeasured", "tbd")
ASSUMED = ("commanded", "assumed")


def limits_from_cfg(cfg):
    lim = P.Limits()
    for key in ("cruise_vx", "cruise_wz", "v_cal_rel_err", "ramp_s",
                "min_primitive_s", "max_total_s", "max_total_m",
                "yaw_tol_deg", "turn_timeout_factor", "settle_s"):
        if key in cfg:
            setattr(lim, key, float(cfg[key]))
    if "max_prims" in cfg:
        lim.max_prims = int(cfg["max_prims"])
    # v_cal has three states and each one means something different:
    #   a number     measured ground speed at cruise_vx; distances carry +-rel_err
    #   commanded    ground speed ASSUMED equal to the command (decision
    #                2026-09-28: no venue to measure, and the task does not need
    #                precise distance). Distances are allowed and labelled as such.
    #   unmeasured   distance commands are refused.
    # Absent means unmeasured, never commanded: a config must say the assumption
    # out loud, or "10 m" would turn into a number that looks measured.
    raw = str(cfg.get("v_cal", "")).strip().lower()
    lim.v_cal_assumed = raw in ASSUMED
    lim.v_cal = None if (raw in UNMEASURED or lim.v_cal_assumed) else float(raw)
    return lim


# --------------------------------------------------------------------- dry run
def dry_run(prims, limits, hz):
    """Execute against a simulated robot that tracks wz perfectly. Prints the
    command trace. No ROS, so this is what you run before you go near the robot."""
    e = ex.Executor(prims, limits)
    dt, t, acc = 1.0 / hz, 0.0, 0.0
    print("\n{:>8}  {:>6} {:>6}  {:<10} {}".format("t", "vx", "wz", "state", "event"))
    last = None
    while t < limits.max_total_s * 2 + 30:
        out = e.step(t, yaw=acc, bridge_state="RUNNING")
        line = (out.vx, out.wz, out.state)
        if out.event or line != last:
            print("{:8.2f}  {:6.3f} {:6.3f}  {:<10} {}".format(
                t, out.vx, out.wz, out.state, out.event or ""))
            last = line
        if out.state in (ex.DONE, ex.ABORTED):
            break
        acc += out.wz * dt
        t += dt
    print("\nreport:\n" + json.dumps(e.report(), indent=2, ensure_ascii=False))
    return 0 if e.state == ex.DONE else 1


# ------------------------------------------------------------------- live run
def live_run(prims, limits, hz, allow_second_writer=False):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float32MultiArray, String

    class Mission(Node):
        def __init__(self):
            Node.__init__(self, "r1_mission")
            self.pub = self.create_publisher(Float32MultiArray, CMD_TOPIC, 10)
            # The bridge publishes its debug topics BEST_EFFORT (KeepLast(1)). A
            # RELIABLE subscription does not match a best-effort publisher, so with
            # the default profile this subscription silently receives nothing and
            # every closed-loop turn aborts with "no IMU yaw".
            self.create_subscription(
                Float32MultiArray, IMU_TOPIC, self.on_imu,
                QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT))
            # The bridge publishes ~/status reliable + transient_local, once a
            # second. Matching that durability delivers the latched last value on
            # connect instead of up to a second of "unknown".
            self.create_subscription(
                String, STATUS_TOPIC, self.on_status,
                QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL))
            self.tracker = Y.YawTracker()
            self.bridge_state = "UNKNOWN"
            self.last_status = None
            self.ex = ex.Executor(prims, limits)
            self.t0 = None
            self.msg = Float32MultiArray()
            self.create_timer(1.0 / hz, self.tick)

        def on_imu(self, m):
            if len(m.data) >= 4:
                self.tracker.update(m.data[0:4])

        def on_status(self, m):
            # The bridge's status line starts with its state name.
            parts = str(m.data).split()
            if parts:
                self.bridge_state = parts[0]
                self.last_status = time.monotonic()

        def send(self, vx, vy, wz):
            self.msg.data = [float(vx), float(vy), float(wz)]
            self.pub.publish(self.msg)

        def tick(self):
            now = time.monotonic()
            if self.t0 is None:
                self.t0 = now
            t = now - self.t0
            yawv = self.tracker.total if self.tracker.ready else None
            state = self.bridge_state
            # 1 Hz publisher: three silent seconds means the bridge is gone or
            # this process has lost it. Either way, stop trusting the last value.
            if self.last_status is not None and now - self.last_status > 3.0:
                state = "NO_STATUS"
            out = self.ex.step(t, yaw=yawv, bridge_state=state)
            self.send(out.vx, out.vy, out.wz)
            if out.event:
                self.get_logger().info(out.event)
            if out.state in (ex.DONE, ex.ABORTED):
                self.finished = True

    # Take SIGINT ourselves. From humble on, rclpy's own handler shuts the context
    # down on Ctrl-C, after which no publish succeeds -- so the "send zeros on the
    # way out" below silently did nothing and the robot kept its last command
    # until the bridge's 500 ms deadman caught it (measured against a stand-in:
    # last cmd_vel after SIGINT was vx=0.30). Foxy has no signal_handler_options,
    # hence the fallback; there the deadman remains the backstop.
    import signal
    stop = {"flag": False}
    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):
        rclpy.init()
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    node = Mission()
    node.finished = False
    if not allow_second_writer and hasattr(node, "count_publishers"):
        # One writer on cmd_vel at a time. Two schedulers fighting over the same
        # topic is indistinguishable, from the robot's side, from a policy that
        # cannot hold a command.
        others = node.count_publishers(CMD_TOPIC) - 1
        if others > 0:
            node.get_logger().error(
                "{} other publisher(s) already on {} -- refusing to be a second "
                "writer. Stop the other one, or pass --allow-second-writer if you "
                "are certain.".format(others, CMD_TOPIC))
            node.destroy_node()
            rclpy.shutdown()
            return 2
    print("waiting for the bridge... (it must reach RUNNING before the plan starts)")
    while rclpy.ok() and not stop["flag"] and not node.finished:
        rclpy.spin_once(node, timeout_sec=0.05)
    if stop["flag"] and node.ex.state not in (ex.DONE, ex.ABORTED):
        node.ex._abort("operator interrupt (Ctrl-C)", 0.0)
        node.get_logger().warn("interrupted -- commanding zero")
    # Zero, several times, and give the executor a chance to flush them. The
    # bridge's deadman is the backstop if this process is killed harder than this.
    try:
        for _ in range(5):
            node.send(0.0, 0.0, 0.0)
            rclpy.spin_once(node, timeout_sec=0.02)
    except Exception as exc:
        print("could not send the final zero ({}); the bridge deadman will "
              "stop the robot within 0.5 s".format(exc))
    rep = node.ex.report()
    node.destroy_node()
    rclpy.shutdown()
    print("\nreport:\n" + json.dumps(rep, indent=2, ensure_ascii=False))
    return 0 if rep["final_state"] == ex.DONE else 1


# ------------------------------------------------------------------------ CLI
SCHEMA = {
    "type": "object",
    "required": ["plan"],
    "properties": {
        "understood": {"type": "string",
                       "description": "restate the instruction you are executing"},
        "plan": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["op"],
                "properties": {
                    "op": {"enum": list(P.OPS)},
                    "vx": {"type": "number", "description": "forward speed, m/s"},
                    "wz": {"type": "number", "description": "turn rate magnitude, rad/s"},
                    "direction": {"enum": ["left", "right"]},
                    "duration_s": {"type": "number"},
                    "distance_m": {"type": "number",
                                   "description": "OPEN LOOP: converted to time "
                                                  "using a calibrated speed, and "
                                                  "reported with an error bar"},
                    "yaw_deg": {"type": "number",
                                "description": "CLOSED LOOP on the IMU heading"},
                },
                "additionalProperties": False,
            },
        },
        "refused": {"type": "array", "items": {"type": "string"},
                    "description": "parts of the instruction you could not express"},
    },
    "additionalProperties": False,
}


def capability_text(env, lim):
    """The capability statement, in English: this is exactly what the LLM layer
    will be given, so it is generated from the enforced envelope and the measured
    calibration rather than written by hand."""
    lines = [env.describe("en")]
    if lim.v_cal_assumed:
        lines.append("I can walk a given distance, but only approximately: I have no "
                     "odometry, so a distance is converted to a walking time at the "
                     "commanded speed, and my real walking speed has not been "
                     "measured.")
    elif lim.v_cal is None:
        lines.append("I can walk for a given time. I cannot walk a given distance yet: "
                     "my walking speed has not been calibrated.")
    else:
        lines.append("I can walk a given distance, but only approximately: I have no "
                     "odometry, so distance is time x a measured speed of {:.2f} m/s, "
                     "accurate to about +-{:.0f}%.".format(lim.v_cal, lim.v_cal_rel_err * 100))
    lines.append("I can turn to a given angle using my IMU heading, to within about "
                 "{:.0f} degrees.".format(lim.yaw_tol_deg))
    lines.append("Each step must last at least {:.0f} s. A plan may have at most {} steps, "
                 "{:.0f} s in total, and {:.0f} m in total."
                 .format(lim.min_primitive_s, lim.max_prims, lim.max_total_s, lim.max_total_m))
    lines.append("Units are metres, seconds, degrees, m/s and rad/s.")
    return " ".join(lines)


def print_capability(env, lim, as_json=False):
    if as_json:
        print(json.dumps({"capability": capability_text(env, lim),
                          "envelope": {"vx": list(env.vx), "vy": list(env.vy),
                                       "wz": list(env.wz), "source": env.source},
                          "v_cal": lim.v_cal, "v_cal_assumed": lim.v_cal_assumed,
                          "schema": SCHEMA},
                         indent=2, ensure_ascii=False))
        return
    print("Capability statement (generated from the deployed configuration; this is")
    print("what the LLM layer is given):\n")
    print("  " + capability_text(env, lim))
    print("\n  envelope source: " + env.source)
    print("  (operator view / 中文: " + env.describe("zh") + ")")
    print("\nJSON schema (structured output target):")
    print(json.dumps(SCHEMA, indent=2, ensure_ascii=False))


def build_steps(args):
    if args.cmd == "run":
        if args.file:
            return P.parse_script(pathlib.Path(args.file).read_text())
        if args.script:
            return P.parse_script(args.script)
        raise P.PlanError("run needs a script string or --file")
    if args.cmd == "json":
        text = (pathlib.Path(args.file).read_text() if args.file
                else sys.stdin.read())
        return json.loads(text)
    if args.cmd == "walk":
        st = {"op": "walk", "vx": args.speed, "text": "walk (cli)"}
        if args.meters is not None:
            st["distance_m"] = args.meters
        else:
            st["duration_s"] = args.seconds
        return [st]
    if args.cmd == "turn":
        st = {"op": "turn", "wz": args.rate, "text": "turn (cli)",
              "direction": "right" if args.right else "left"}
        if args.seconds is not None:
            st["duration_s"] = args.seconds
        else:
            st["yaw_deg"] = args.deg
        return [st]
    if args.cmd == "stand":
        return [{"op": "stand", "duration_s": args.seconds, "text": "stand (cli)"}]
    if args.cmd == "stop":
        return [{"op": "stop", "text": "stop (cli)"}]
    raise P.PlanError("unknown command " + str(args.cmd))


def main(argv=None):
    # The shared flags go on BOTH the top-level parser and every subcommand, so
    # `run ... --dry-run` works as naturally as `--dry-run run ...`. Getting this
    # wrong means an operator who types --dry-run in the obvious place gets an
    # argparse error next to a robot that is about to move.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=str(DEFAULT_CFG))
    common.add_argument("--envelope", default=None,
                        help="command_envelope.json (default: the installed bundle's)")
    common.add_argument("--pub-hz", type=float, default=10.0,
                        help="cmd_vel publish rate (must beat the 500 ms deadman)")
    common.add_argument("--dry-run", action="store_true",
                        help="simulate, print the command trace, touch no robot")
    common.add_argument("--yes", action="store_true", help="skip the confirmation")
    common.add_argument("--allow-second-writer", action="store_true")

    ap = argparse.ArgumentParser(
        prog="r1_mission", parents=[common],
        description="Drive the R1 by time / angle / speed over the bridge's "
                    "existing topics. One process per plan.")
    sub = ap.add_subparsers(dest="cmd")

    w = sub.add_parser("walk", parents=[common],
                       help="walk for a time, or an (open-loop) distance")
    w.add_argument("--seconds", type=float, default=5.0)
    w.add_argument("--meters", type=float, default=None)
    w.add_argument("--speed", type=float, default=None)

    t = sub.add_parser("turn", parents=[common],
                       help="turn by an angle (closed loop) or a time")
    t.add_argument("--deg", type=float, default=90.0)
    t.add_argument("--seconds", type=float, default=None)
    t.add_argument("--rate", type=float, default=None, help="|wz| in rad/s")
    g = t.add_mutually_exclusive_group()
    g.add_argument("--left", action="store_true")
    g.add_argument("--right", action="store_true")

    st = sub.add_parser("stand", parents=[common], help="hold zero velocity")
    st.add_argument("--seconds", type=float, default=5.0)

    sub.add_parser("stop", parents=[common], help="command zero and exit")
    cp = sub.add_parser("capability", parents=[common],
                        help="what the robot can do, and the JSON schema")
    cp.add_argument("--json", action="store_true",
                    help="machine-readable: capability text + envelope + schema")

    r = sub.add_parser("run", parents=[common],
                       help='a script: "walk 5s@0.4; turn left 90; walk 10m"')
    r.add_argument("script", nargs="?", default=None)
    r.add_argument("--file", default=None)

    j = sub.add_parser("json", parents=[common],
                       help="a JSON plan on stdin or --file (the LLM path)")
    j.add_argument("--file", default=None)

    args = ap.parse_args(argv)
    if not args.cmd:
        ap.print_help()
        return 2

    try:
        cfg = read_flat_yaml(args.config) if pathlib.Path(args.config).is_file() else {}
        lim = limits_from_cfg(cfg)
        env = resolve_envelope(args.envelope)
        if getattr(args, "speed", None) is None and args.cmd == "walk":
            args.speed = lim.cruise_vx
        if getattr(args, "rate", None) is None and args.cmd == "turn":
            args.rate = lim.cruise_wz

        if args.cmd == "capability":
            print_capability(env, lim, as_json=args.json)
            return 0

        raw = build_steps(args)
        if args.cmd == "json":
            prims, tot_s, tot_m = P.plan_from_json(raw, env, lim)
        else:
            prims, tot_s, tot_m = P.compile_plan(raw, env, lim)
    except P.PlanError as exc:
        print("\nREFUSED: {}".format(exc), file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print("\nREFUSED: {}".format(exc), file=sys.stderr)
        return 2

    print("plan ({} primitives, ~{:.1f}s, ~{:.1f}m):".format(len(prims), tot_s, tot_m))
    for i, pr in enumerate(prims):
        print("  {}. {}".format(i + 1, pr.summary()))
    print("envelope: {}  [{}]".format(env.describe(), env.source))
    if any(pr.distance_m is not None for pr in prims):
        print("NOTE: every distance here is OPEN LOOP (time x {}). There is no "
              "odometry on this robot.".format(
                  "the COMMANDED speed, assumed and not measured" if lim.v_cal_assumed
                  else "calibrated speed"))

    if args.dry_run:
        return dry_run(prims, lim, args.pub_hz)

    if not args.yes:
        try:
            if input("\nexecute on the robot? [y/N] ").strip().lower() not in ("y", "yes"):
                print("not executed.")
                return 2
        except EOFError:
            print("\nnot executed (no tty; pass --yes).")
            return 2
    return live_run(prims, lim, args.pub_hz, args.allow_second_writer)
