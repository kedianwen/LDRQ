"""Primitives, the script language, and the checks that run before anything moves.

No ROS, no numpy: importable and testable on any box, and Python 3.8 clean
because the robot is ROS 2 foxy on Ubuntu 20.04.

The asymmetry this module exists to make explicit:

  * turn(angle)   CAN be closed-loop. The bridge publishes the IMU quaternion on
                  ~/imu at 50 Hz, so yaw is measurable and the primitive ends
                  when the robot has actually turned.
  * walk(distance) CANNOT. There is no base linear velocity in the 425-dim
                  observation and no odometry topic anywhere, so distance is
                  time x a CALIBRATED speed and carries an error bar. The plan
                  keeps that error bar and reports it; it never prints a bare
                  number that looks measured.
"""
from __future__ import annotations

import json
import math
import re

OPS = ("walk", "turn", "stand", "stop")


class PlanError(Exception):
    """A specific, actionable reason a plan will not be executed."""


class Envelope(object):
    """The command envelope AS TRAINED. Not a comfort limit.

    An axis pinned to [0, 0] is refused rather than clamped: it was pinned for
    all of training, so a non-zero value there is not a small extrapolation, it
    is a state that does not exist in the distribution.
    """

    def __init__(self, vx, vy, wz, source="defaults"):
        self.vx = (float(vx[0]), float(vx[1]))
        self.vy = (float(vy[0]), float(vy[1]))
        self.wz = (float(wz[0]), float(wz[1]))
        self.source = source

    @classmethod
    def from_bundle(cls, path):
        d = json.loads(open(str(path)).read())
        return cls(d["vx"], d["vy"], d["wz"], source=str(path))

    @classmethod
    def from_bridge_yaml(cls, path):
        """Parse the three cmd_vel_*_range lines. Deliberately not a yaml parse:
        the robot may not have PyYAML, and three lines do not need one."""
        text = open(str(path)).read()

        def rng(key):
            m = re.search(re.escape(key) + r"\s*:\s*\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\]",
                          text)
            if not m:
                raise PlanError("{}: no '{}' line".format(path, key))
            return (float(m.group(1)), float(m.group(2)))

        return cls(rng("cmd_vel_x_range"), rng("cmd_vel_y_range"),
                   rng("cmd_vel_yaw_range"), source=str(path))

    def pinned(self, axis):
        lo, hi = getattr(self, axis)
        return lo == 0.0 and hi == 0.0

    def check(self, axis, value, what):
        lo, hi = getattr(self, axis)
        if self.pinned(axis):
            raise PlanError(
                "{}: {}={:.3f} but this axis was pinned to 0 for ALL of training. "
                "It is refused, not clamped -- the policy has never seen a "
                "non-zero value here.".format(what, axis, value))
        if not (lo - 1e-9 <= value <= hi + 1e-9):
            raise PlanError("{}: {}={:.3f} is outside the trained range "
                            "[{:.2f}, {:.2f}] ({})"
                            .format(what, axis, value, lo, hi, self.source))

    def describe(self):
        """The prose the robot uses to say what it can do. Rendered from the same
        numbers the bridge clamps to, so it cannot drift out of step with them --
        this is also what the LLM layer will send as its capability statement."""
        bits = []
        lo, hi = self.vx
        bits.append("前进 {:.2f}~{:.2f} m/s{}".format(lo, hi,
                    "（不能倒车）" if lo >= 0 else ""))
        if self.pinned("vy"):
            bits.append("不能横移（训练中该轴全程钉死为 0）")
        else:
            bits.append("横移 {:.2f}~{:.2f} m/s".format(*self.vy))
        lo, hi = self.wz
        bits.append("原地/行进转向 {:.2f}~{:.2f} rad/s".format(lo, hi))
        return "；".join(bits)


class Prim(object):
    """One resolved primitive: a constant (vx, wz) plus how it ends."""

    def __init__(self, op, vx=0.0, wz=0.0, duration_s=None, yaw_deg=None,
                 distance_m=None, distance_err_m=None, text=""):
        self.op = op
        self.vx = float(vx)
        self.wz = float(wz)
        self.duration_s = None if duration_s is None else float(duration_s)
        self.yaw_deg = None if yaw_deg is None else float(yaw_deg)
        self.distance_m = None if distance_m is None else float(distance_m)
        self.distance_err_m = None if distance_err_m is None else float(distance_err_m)
        self.text = text

    @property
    def closed_loop(self):
        return self.op == "turn" and self.yaw_deg is not None

    def __repr__(self):
        return "Prim({})".format(self.summary())

    def summary(self):
        if self.op == "walk":
            s = "walk vx={:.2f} for {:.2f}s".format(self.vx, self.duration_s or 0.0)
            if self.distance_m is not None:
                s += " (~{:.2f}±{:.2f} m)".format(self.distance_m,
                                                  self.distance_err_m or 0.0)
            return s
        if self.op == "turn":
            if self.closed_loop:
                return "turn {:+.1f}deg wz={:+.2f} (closed loop on IMU yaw)".format(
                    self.yaw_deg, self.wz)
            return "turn wz={:+.2f} for {:.2f}s (open loop)".format(
                self.wz, self.duration_s or 0.0)
        if self.op == "stand":
            return "stand for {:.2f}s".format(self.duration_s or 0.0)
        return "stop"


class Limits(object):
    """Execution-side limits. These are policy-safety choices, not physics.

    min_primitive_s is the interesting one. Training resampled the velocity
    command every 10 s (resampling_time_range=(10.0, 10.0)), so a step change is
    in distribution but a sequence of half-second primitives puts the robot in a
    transient it saw rarely. 2 s is a deliberate floor, with that as the reason.
    """

    def __init__(self, cruise_vx=0.4, cruise_wz=0.4, v_cal=None, v_cal_rel_err=0.15,
                 ramp_s=0.5, min_primitive_s=2.0, max_total_s=120.0,
                 max_total_m=30.0, max_prims=20, yaw_tol_deg=3.0,
                 turn_timeout_factor=2.0, settle_s=1.0):
        self.cruise_vx = float(cruise_vx)
        self.cruise_wz = float(cruise_wz)
        self.v_cal = None if v_cal is None else float(v_cal)
        self.v_cal_rel_err = float(v_cal_rel_err)
        self.ramp_s = float(ramp_s)
        self.min_primitive_s = float(min_primitive_s)
        self.max_total_s = float(max_total_s)
        self.max_total_m = float(max_total_m)
        self.max_prims = int(max_prims)
        self.yaw_tol_deg = float(yaw_tol_deg)
        self.turn_timeout_factor = float(turn_timeout_factor)
        self.settle_s = float(settle_s)


_WALK = re.compile(r"^walk\s+([\d.]+)\s*(s|m)\s*(?:@\s*([\d.]+))?$", re.I)
_TURN_DEG = re.compile(r"^turn\s+(left|right)\s+([\d.]+)\s*(?:deg)?\s*(?:@\s*([\d.]+))?$", re.I)
_TURN_S = re.compile(r"^turn\s+(left|right)\s+([\d.]+)\s*s\s*(?:@\s*([\d.]+))?$", re.I)
_STAND = re.compile(r"^stand\s+([\d.]+)\s*s$", re.I)


def parse_script(script):
    """`"walk 5s@0.4; turn left 90; walk 10m; stand 3s"` -> list of raw steps.

    Kept as a tiny language rather than json for the CLI, because the operator is
    typing it in a terminal next to a robot. compile_plan() then does all the
    checking, so this and the LLM's json converge on the same Prim list.
    """
    steps = []
    for raw in re.split(r"[;\n]", script):
        item = raw.strip()
        if not item or item.startswith("#"):
            continue
        low = item.lower()
        m = _TURN_S.match(item)
        if m:
            steps.append({"op": "turn", "direction": m.group(1).lower(),
                          "duration_s": float(m.group(2)),
                          "wz": float(m.group(3)) if m.group(3) else None,
                          "text": item})
            continue
        m = _TURN_DEG.match(item)
        if m:
            steps.append({"op": "turn", "direction": m.group(1).lower(),
                          "yaw_deg": float(m.group(2)),
                          "wz": float(m.group(3)) if m.group(3) else None,
                          "text": item})
            continue
        m = _WALK.match(item)
        if m:
            key = "duration_s" if m.group(2).lower() == "s" else "distance_m"
            steps.append({"op": "walk", key: float(m.group(1)),
                          "vx": float(m.group(3)) if m.group(3) else None,
                          "text": item})
            continue
        m = _STAND.match(item)
        if m:
            steps.append({"op": "stand", "duration_s": float(m.group(1)),
                          "text": item})
            continue
        if low == "stop":
            steps.append({"op": "stop", "text": item})
            continue
        raise PlanError(
            "cannot parse '{}'.\n"
            "  walk <N>s[@<v>] | walk <N>m[@<v>] | turn left|right <deg>[@<wz>] |\n"
            "  turn left|right <N>s[@<wz>] | stand <N>s | stop".format(item))
    if not steps:
        raise PlanError("empty plan")
    return steps


def compile_plan(steps, env, limits):
    """Raw steps -> validated Prim list. Every refusal in here happens BEFORE a
    single cmd_vel message is published."""
    if len(steps) > limits.max_prims:
        raise PlanError("{} primitives exceeds the budget of {}"
                        .format(len(steps), limits.max_prims))

    prims, total_s, total_m = [], 0.0, 0.0
    for i, st in enumerate(steps):
        what = "step {} ({})".format(i + 1, st.get("text") or st["op"])
        op = st["op"]
        if op not in OPS:
            raise PlanError("{}: unknown op '{}'".format(what, op))

        if op == "stop":
            prims.append(Prim("stop", text=st.get("text", "stop")))
            continue

        if op == "stand":
            dur = float(st["duration_s"])
            if dur < limits.min_primitive_s:
                raise PlanError(_too_short(what, dur, limits))
            prims.append(Prim("stand", duration_s=dur, text=st.get("text", "")))
            total_s += dur
            continue

        if op == "walk":
            vx = limits.cruise_vx if st.get("vx") is None else float(st["vx"])
            env.check("vx", vx, what)
            if vx <= 0.0:
                raise PlanError("{}: walking needs vx > 0".format(what))
            dist = st.get("distance_m")
            err = None
            if dist is not None:
                if limits.v_cal is None:
                    raise PlanError(
                        "{}: asked for a DISTANCE, but no calibrated speed is "
                        "configured.\n"
                        "  There is no base linear velocity in the observation and no\n"
                        "  odometry topic, so distance can only be time x a measured\n"
                        "  speed. Measure it (tape + stopwatch) and set v_cal in\n"
                        "  mission.yaml, or ask for a duration instead.".format(what))
                # The commanded vx is what the policy is asked for; v_cal is what
                # the robot actually achieves at that command. Scale the measured
                # speed by the command ratio rather than pretending they are equal.
                achieved = limits.v_cal * (vx / limits.cruise_vx)
                dur = float(dist) / achieved
                err = float(dist) * limits.v_cal_rel_err
                total_m += float(dist)
            else:
                dur = float(st["duration_s"])
                if limits.v_cal is not None:
                    achieved = limits.v_cal * (vx / limits.cruise_vx)
                    dist = achieved * dur
                    err = dist * limits.v_cal_rel_err
                    total_m += dist
            if dur < limits.min_primitive_s:
                raise PlanError(_too_short(what, dur, limits))
            prims.append(Prim("walk", vx=vx, duration_s=dur, distance_m=dist,
                              distance_err_m=err, text=st.get("text", "")))
            total_s += dur
            continue

        # turn
        sign = -1.0 if st.get("direction") == "right" else 1.0
        mag = limits.cruise_wz if st.get("wz") is None else abs(float(st["wz"]))
        wz = sign * mag
        env.check("wz", wz, what)
        if mag <= 0.0:
            raise PlanError("{}: turning needs |wz| > 0".format(what))
        if st.get("yaw_deg") is not None:
            deg = sign * abs(float(st["yaw_deg"]))
            dur = math.radians(abs(deg)) / mag
            if dur < limits.min_primitive_s:
                raise PlanError(
                    "{}: {:.0f} deg at {:.2f} rad/s takes only {:.2f}s, under the "
                    "{:.1f}s floor. Turn slower (@ a smaller wz) or accept a longer "
                    "turn.".format(what, abs(deg), mag, dur, limits.min_primitive_s))
            prims.append(Prim("turn", wz=wz, yaw_deg=deg, duration_s=dur,
                              text=st.get("text", "")))
        else:
            dur = float(st["duration_s"])
            if dur < limits.min_primitive_s:
                raise PlanError(_too_short(what, dur, limits))
            prims.append(Prim("turn", wz=wz, duration_s=dur, text=st.get("text", "")))
        total_s += dur

    if total_s > limits.max_total_s:
        raise PlanError("total {:.1f}s exceeds the budget of {:.0f}s"
                        .format(total_s, limits.max_total_s))
    if total_m > limits.max_total_m:
        raise PlanError("total ~{:.1f}m exceeds the budget of {:.0f}m"
                        .format(total_m, limits.max_total_m))
    return prims, total_s, total_m


def _too_short(what, dur, limits):
    return ("{}: {:.2f}s is under the {:.1f}s floor.\n"
            "  Training resampled the velocity command every 10 s, so very short\n"
            "  primitives chain together into a transient the policy saw rarely."
            .format(what, dur, limits.min_primitive_s))


def plan_from_json(obj, env, limits):
    """The LLM layer's entry point: the same compiler, a json front end.

    Untrusted input by construction, so the shape is checked here and the values
    are checked by compile_plan -- the model never gets to widen the envelope.
    """
    if isinstance(obj, (str, bytes)):
        try:
            obj = json.loads(obj)
        except ValueError as exc:
            raise PlanError("not valid JSON: {}".format(exc))
    if not isinstance(obj, dict) or not isinstance(obj.get("plan"), list):
        raise PlanError("expected an object with a 'plan' array")
    steps = []
    for i, item in enumerate(obj["plan"]):
        if not isinstance(item, dict):
            raise PlanError("plan[{}] is not an object".format(i))
        op = item.get("op")
        if op not in OPS:
            raise PlanError("plan[{}]: op must be one of {}, got {!r}"
                            .format(i, ", ".join(OPS), op))
        unknown = set(item) - {"op", "vx", "wz", "duration_s", "distance_m",
                               "yaw_deg", "direction", "text"}
        if unknown:
            raise PlanError("plan[{}]: unknown field(s) {}"
                            .format(i, ", ".join(sorted(unknown))))
        if op == "walk" and (("duration_s" in item) == ("distance_m" in item)):
            raise PlanError("plan[{}]: walk needs exactly one of duration_s or "
                            "distance_m".format(i))
        if op == "turn":
            if item.get("direction") not in ("left", "right"):
                raise PlanError("plan[{}]: turn needs direction 'left' or 'right'"
                                .format(i))
            if ("yaw_deg" in item) == ("duration_s" in item):
                raise PlanError("plan[{}]: turn needs exactly one of yaw_deg or "
                                "duration_s".format(i))
        st = dict(item)
        st["text"] = item.get("text") or "{} (from json)".format(op)
        steps.append(st)
    return compile_plan(steps, env, limits)
