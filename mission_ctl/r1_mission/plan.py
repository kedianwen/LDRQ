"""Primitives, the script language, and the checks that run before anything moves.

No ROS, no numpy: importable and testable on any box, and Python 3.8 clean
because the robot is ROS 2 foxy on Ubuntu 20.04.

The asymmetry this module exists to make explicit:

  * turn(angle)   CAN be closed-loop. The bridge publishes the IMU quaternion on
                  ~/imu at 50 Hz, so yaw is measurable and the primitive ends
                  when the robot has actually turned.
  * walk(distance) CANNOT. There is no base linear velocity in the 425-dim
                  observation and no odometry topic anywhere, so distance is
                  time x a speed. That speed is either CALIBRATED (and then
                  carries an error bar) or ASSUMED equal to the command (and
                  then says so). Either way the plan never prints a bare number
                  that looks measured.
"""
from __future__ import annotations

import json
import math
import re

OPS = ("walk", "turn", "stand", "stop")

# The closed-loop turn runs at the commanded rate until TAPER_DEG from the target,
# then slows in proportion to what is left, never below TAPER_FLOOR of the rate
# (executor.py). Any estimate of how long a turn takes has to include that: a
# 180 deg turn at 0.4 rad/s is 7.85 s at the bare rate but 9.25 s as executed, so a
# "2x" timeout on the bare rate was really 1.7x -- and on the gantry four of ten
# sweep points hit it (2026-09-28).
TAPER_DEG = 25.0
TAPER_FLOOR = 0.35


def taper_floor(wz, min_wz=0.0):
    """Fraction of |wz| the taper never goes below: TAPER_FLOOR, raised so the
    commanded rate stays at or above min_wz (the robot's measured turning deadband)."""
    m = abs(wz)
    if m <= 0.0:
        return TAPER_FLOOR
    return min(1.0, max(TAPER_FLOOR, min_wz / m))


def expected_turn_s(yaw_deg, wz, ramp_s, min_wz=0.0):
    """How long the executor takes to turn yaw_deg at |wz| on a robot that tracks
    the command exactly: half the ramp, the full-rate part, the proportional
    taper, and the last stretch at the floor rate."""
    a = math.radians(abs(yaw_deg))
    m = abs(wz)
    if a <= 0.0 or m <= 0.0:
        return 0.0
    f = taper_floor(m, min_wz)
    band = math.radians(TAPER_DEG)
    floor_band = f * band
    t = max(0.0, ramp_s) / 2.0
    t += max(0.0, a - band) / m                                   # full rate
    if a > floor_band:                                            # w ~ remaining
        t += (band / m) * math.log(min(a, band) / floor_band)
    t += min(a, floor_band) / (f * m)                             # floor rate
    return t


def achieved_speed(vx, limits):
    """The ground speed a commanded vx is taken to produce, or None if there is
    nothing to convert a distance with."""
    if limits.v_cal_assumed:
        # Decided 2026-09-28: no venue to measure it, and the task only needs
        # distances at the level of "a few metres". Taken as equal to the command,
        # and reported as an assumption, never with an error bar that looks measured.
        return vx
    if limits.v_cal is None:
        return None
    # The commanded vx is what the policy is asked for; v_cal is what the robot
    # actually achieves at cruise_vx. Scale by the command ratio.
    return limits.v_cal * (vx / limits.cruise_vx)


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

    def describe(self, lang="en"):
        """What the robot can do, rendered from the same numbers the bridge clamps
        to, so it cannot drift out of step with them.

        English by default: the LLM layer takes English, and its refusal messages
        (nl.refusal_message) are built from the same numbers. The model itself is
        not given this -- it transcribes, the robot judges. `lang="zh"` renders the
        same facts for an operator.
        """
        vlo, vhi = self.vx
        wlo, whi = self.wz
        if lang == "zh":
            bits = ["前进 {:.2f}~{:.2f} m/s{}".format(vlo, vhi,
                                                   "（不能倒车）" if vlo >= 0 else "")]
            bits.append("不能横移（训练中该轴全程钉死为 0）" if self.pinned("vy")
                        else "横移 {:.2f}~{:.2f} m/s".format(*self.vy))
            bits.append("原地转向，最快 {:.2f} rad/s（边走边转未验证）".format(max(abs(wlo), abs(whi))))
            return "；".join(bits)
        bits = ["I can walk forward at {:.2f} to {:.2f} m/s{}".format(
            vlo, vhi, " (I cannot walk backward)" if vlo >= 0 else "")]
        bits.append("I cannot move sideways (that axis was fixed at zero for all "
                    "of my training)" if self.pinned("vy")
                    else "I can move sideways at {:.2f} to {:.2f} m/s".format(*self.vy))
        # On the spot only: mission_ctl has no walk-and-turn primitive, and on the PG-2
        # run (2026-09-29) a 0.15 rad/s turn while walking was lost entirely.
        bits.append("I can turn left or right on the spot at up to {:.2f} rad/s (I do not "
                    "turn while walking)".format(max(abs(wlo), abs(whi))))
        return "; ".join(bits) + "."


class Prim(object):
    """One resolved primitive: a constant (vx, wz) plus how it ends."""

    def __init__(self, op, vx=0.0, wz=0.0, duration_s=None, yaw_deg=None,
                 distance_m=None, distance_err_m=None, text="", expected_s=None):
        self.op = op
        self.vx = float(vx)
        self.wz = float(wz)
        self.duration_s = None if duration_s is None else float(duration_s)
        self.yaw_deg = None if yaw_deg is None else float(yaw_deg)
        self.distance_m = None if distance_m is None else float(distance_m)
        self.distance_err_m = None if distance_err_m is None else float(distance_err_m)
        self.text = text
        # closed-loop turns: how long it should take as executed (ramp + taper).
        # duration_s stays the bare angle / rate, which is what the report calls
        # commanded_s; the timeout and the plan total use this one.
        self.expected_s = None if expected_s is None else float(expected_s)

    @property
    def closed_loop(self):
        return self.op == "turn" and self.yaw_deg is not None

    def __repr__(self):
        return "Prim({})".format(self.summary())

    def summary(self):
        if self.op == "walk":
            s = "walk vx={:.2f} for {:.2f}s".format(self.vx, self.duration_s or 0.0)
            if self.distance_m is not None:
                if self.distance_err_m is None:
                    s += " (~{:.2f} m if speed = command; not measured)".format(
                        self.distance_m)
                else:
                    s += " (~{:.2f}±{:.2f} m)".format(self.distance_m,
                                                      self.distance_err_m)
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
                 turn_timeout_factor=2.0, settle_s=1.0, start_timeout_s=10.0,
                 v_cal_assumed=False, turn_min_wz=0.0):
        self.cruise_vx = float(cruise_vx)
        self.cruise_wz = float(cruise_wz)
        self.v_cal = None if v_cal is None else float(v_cal)
        self.v_cal_rel_err = float(v_cal_rel_err)
        # True: ground speed is taken to equal the command (not measured). Wins
        # over v_cal, so a config cannot be half one and half the other.
        self.v_cal_assumed = bool(v_cal_assumed)
        # Smallest |wz| the real robot actually turns at (turn_response.py). 0 = not
        # measured: behaviour as before. The PG-2 run turned at 0.00 of a 0.15 command.
        self.turn_min_wz = float(turn_min_wz)
        self.ramp_s = float(ramp_s)
        self.min_primitive_s = float(min_primitive_s)
        self.max_total_s = float(max_total_s)
        self.max_total_m = float(max_total_m)
        self.max_prims = int(max_prims)
        self.yaw_tol_deg = float(yaw_tol_deg)
        self.turn_timeout_factor = float(turn_timeout_factor)
        self.settle_s = float(settle_s)
        self.start_timeout_s = float(start_timeout_s)


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
    wz_max = max(abs(env.wz[0]), abs(env.wz[1]))
    if limits.turn_min_wz > wz_max + 1e-9:
        # Not a per-step refusal: with this config EVERY closed-loop turn would be
        # refused, which reads like a robot that cannot turn. Say it is the config.
        raise PlanError(
            "config error: turn_min_wz = {:.2f} rad/s is above the trained turn range "
            "(|wz| <= {:.2f}, {}). It is the smallest rate the robot turns at, from "
            "turn_response.py -- not a gain. Fix mission.yaml."
            .format(limits.turn_min_wz, wz_max, env.source))
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
            achieved = achieved_speed(vx, limits)
            if dist is not None:
                if achieved is None:
                    raise PlanError(
                        "{}: asked for a DISTANCE, but no speed is configured to\n"
                        "  convert it with. There is no base linear velocity in the\n"
                        "  observation and no odometry topic, so distance can only be\n"
                        "  time x a speed. Set v_cal in mission.yaml (a measured\n"
                        "  number, or 'commanded' to assume the command), or ask for\n"
                        "  a duration instead.".format(what))
                dur = float(dist) / achieved
                if not limits.v_cal_assumed:
                    err = float(dist) * limits.v_cal_rel_err
                total_m += float(dist)
            else:
                dur = float(st["duration_s"])
                if achieved is not None:
                    dist = achieved * dur
                    if not limits.v_cal_assumed:
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
        if st.get("yaw_deg") is not None and mag < limits.turn_min_wz:
            raise PlanError(
                "{}: {:.2f} rad/s is below turn_min_wz = {:.2f}, the lowest rate this "
                "robot was measured to turn at; slower turns are unmeasured and may "
                "stall. Turn at >= {:.2f}.".format(what, mag, limits.turn_min_wz,
                                                   limits.turn_min_wz))
        if st.get("yaw_deg") is not None:
            deg = sign * abs(float(st["yaw_deg"]))
            dur = math.radians(abs(deg)) / mag
            if dur < limits.min_primitive_s:
                raise PlanError(
                    "{}: {:.0f} deg at {:.2f} rad/s takes only {:.2f}s, under the "
                    "{:.1f}s floor. Turn slower (@ a smaller wz) or accept a longer "
                    "turn.".format(what, abs(deg), mag, dur, limits.min_primitive_s))
            exp = expected_turn_s(deg, mag, limits.ramp_s, limits.turn_min_wz)
            prims.append(Prim("turn", wz=wz, yaw_deg=deg, duration_s=dur,
                              text=st.get("text", ""), expected_s=exp))
            total_s += exp
            continue
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
