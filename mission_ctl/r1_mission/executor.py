"""The behavior state machine. Pure Python: the clock and the measurements are
arguments, so the whole thing is testable without ROS or a robot.

Deliberately NOT in the bridge. The bridge is the only thing that writes
rt/lowcmd, it has been stable since W06, and every change there is a safety
regression risk. Keeping mission logic in a separate process also means it can be
killed without disturbing the bridge's PD hold -- and because the bridge's
cmd_vel deadman decays a stale command to zero in 500 ms, a mission process that
dies stops the robot by construction rather than by remembering to.
"""
from __future__ import annotations

import math

from .plan import TAPER_DEG, taper_floor

IDLE, RUNNING, DONE, ABORTED = "IDLE", "RUNNING", "DONE", "ABORTED"


class Output(object):
    __slots__ = ("vx", "vy", "wz", "state", "event")

    def __init__(self, vx, vy, wz, state, event=None):
        self.vx, self.vy, self.wz = vx, vy, wz
        self.state = state
        self.event = event

    def as_tuple(self):
        return (self.vx, self.vy, self.wz)


class Record(object):
    """What actually happened for one primitive. This is the report the LLM layer
    sends back, so every field is either measured or explicitly an estimate."""

    def __init__(self, prim):
        self.prim = prim
        self.commanded_s = prim.duration_s
        self.expected_s = prim.expected_s
        self.actual_s = None
        self.target_deg = prim.yaw_deg
        self.achieved_deg = None
        self.est_distance_m = prim.distance_m
        self.est_err_m = prim.distance_err_m
        self.aborted = None

    def to_dict(self):
        d = {"op": self.prim.op, "text": self.prim.text}
        if self.commanded_s is not None:
            d["commanded_s"] = round(self.commanded_s, 3)
        if self.expected_s is not None:
            # a closed-loop turn as executed on a robot that tracks exactly; the
            # ratio actual_s / expected_s is what stage B reads (~1.25 on the real robot)
            d["expected_s"] = round(self.expected_s, 3)
        if self.actual_s is not None:
            d["actual_s"] = round(self.actual_s, 3)
        if self.target_deg is not None:
            d["target_deg"] = round(self.target_deg, 2)
        if self.achieved_deg is not None:
            d["achieved_deg"] = round(self.achieved_deg, 2)
            d["source"] = "imu_yaw"
        if self.est_distance_m is not None:
            d["est_distance_m"] = round(self.est_distance_m, 3)
            if self.est_err_m is None:
                d["distance_source"] = ("open loop: commanded time x commanded speed "
                                        "(assumed equal to ground speed, not measured)")
            else:
                d["est_err_m"] = round(self.est_err_m, 3)
                d["distance_source"] = "open loop: commanded time x calibrated speed"
        if self.aborted:
            d["aborted"] = self.aborted
        return d


class Executor(object):
    def __init__(self, prims, limits):
        self.prims = list(prims)
        self.lim = limits
        self.state = IDLE
        self.i = -1
        self.records = []
        self._t_prim = None          # when the current primitive started
        self._from = (0.0, 0.0)      # (vx, wz) we are ramping away from
        self._yaw0 = None
        self._t_settle = None
        self._t_first = None
        self._t_last = None          # the clock at the last step()
        self.abort_reason = None

    # ---------------------------------------------------------------- helpers
    @property
    def current(self):
        return self.prims[self.i] if 0 <= self.i < len(self.prims) else None

    def _target(self, prim):
        if prim is None or prim.op in ("stand", "stop"):
            return (0.0, 0.0)
        return (prim.vx, prim.wz)

    def _ramped(self, t, prim):
        """Linear ramp from the previous target to this one over ramp_s.

        Ramping is a design choice, not a policy limitation: training resampled
        the command every 10 s as a step, so steps are in distribution -- but a
        0.5 s ramp costs nothing and keeps the first control cycles of each
        primitive out of the sharpest transient.
        """
        tgt = self._target(prim)
        if self.lim.ramp_s <= 0.0:
            return tgt
        a = min(1.0, max(0.0, (t - self._t_prim) / self.lim.ramp_s))
        return (self._from[0] + a * (tgt[0] - self._from[0]),
                self._from[1] + a * (tgt[1] - self._from[1]))

    def _advance(self, t):
        prev = self._target(self.current)
        self.i += 1
        self._from = prev
        self._t_prim = t
        self._yaw0 = None
        if self.current is not None:
            self.records.append(Record(self.current))

    # ------------------------------------------------------------------- step
    def step(self, t, yaw=None, bridge_state="RUNNING"):
        """Advance to wall-clock `t`. `yaw` is accumulated radians from
        YawTracker (None if not yet available). Returns an Output."""
        if self.state in (DONE, ABORTED):
            return Output(0.0, 0.0, 0.0, self.state)

        if self._t_first is None:
            self._t_first = t
        self._t_last = t

        # The bridge's own health machine outranks the mission. A DEGRADED bridge
        # is already commanding damping; sending velocity into that is pointless
        # and hides the fault. Once running, anything but RUNNING -- including a
        # status stream that has gone quiet -- ends the plan.
        if self.state == RUNNING and bridge_state != "RUNNING":
            return self._abort("bridge left RUNNING ({})".format(bridge_state), t)

        if self.state == IDLE:
            if bridge_state == "DEGRADED":
                return self._abort("bridge is DEGRADED -- clear it (~/resume) first", t)
            if bridge_state != "RUNNING":
                # Wait, do not start: before the first ~/status arrives the state is
                # simply unknown, and the policy may still be filling its history.
                # Waiting for ever would hang any caller, so it is bounded.
                if t - self._t_first > self.lim.start_timeout_s:
                    why = ("no ~/status received at all -- is the stack up, and is "
                           "this terminal on the same ROS_DOMAIN_ID / "
                           "ROS_LOCALHOST_ONLY / RMW?"
                           if bridge_state in ("UNKNOWN", "NO_STATUS")
                           else "bridge stayed in {}".format(bridge_state))
                    return self._abort("never saw the bridge RUNNING within {:.0f} s: {}"
                                       .format(self.lim.start_timeout_s, why), t)
                return Output(0.0, 0.0, 0.0, IDLE, None)
            self.state = RUNNING
            self._advance(t)
            return Output(0.0, 0.0, 0.0, RUNNING,
                          "start: " + (self.current.summary() if self.current else ""))

        # Once the final primitive is done we are only holding zero. Re-entering
        # the per-primitive logic here would keep rewriting that primitive's
        # record (its actual_s would come out as duration + settle_s).
        if self._t_settle is not None:
            return self._finish(t)

        prim, rec = self.current, self.records[-1] if self.records else None
        if prim is None:
            return self._finish(t)

        elapsed = t - self._t_prim
        finished, note = False, None

        if prim.closed_loop:
            if yaw is None:
                if elapsed > 2.0:
                    return self._abort("no IMU yaw after 2 s -- is ~/imu publishing?", t)
                return Output(0.0, 0.0, 0.0, RUNNING, "waiting for IMU yaw")
            if self._yaw0 is None:
                self._yaw0 = yaw
            turned = yaw - self._yaw0
            want = math.radians(prim.yaw_deg)
            remaining = want - turned
            rec.achieved_deg = math.degrees(turned)
            reached = remaining <= 0.0 if want >= 0.0 else remaining >= 0.0
            if reached:
                finished = True
                err = math.degrees(turned) - prim.yaw_deg
                note = "turn done: {:+.1f} deg (target {:+.1f}, err {:+.1f})".format(
                    math.degrees(turned), prim.yaw_deg, err)
                if abs(err) > self.lim.yaw_tol_deg:
                    rec.aborted = None   # not a failure, but say so in the report
                    note += " -- OUTSIDE the {:.1f} deg tolerance".format(
                        self.lim.yaw_tol_deg)
            elif elapsed > (prim.expected_s or prim.duration_s or 1.0) \
                    * self.lim.turn_timeout_factor:
                rec.aborted = ("turn timed out at {:+.1f} of {:+.1f} deg"
                               .format(math.degrees(turned), prim.yaw_deg))
                return self._abort(rec.aborted, t)
            else:
                # Taper near the target so the turn does not overshoot on the last
                # control cycles, but never below a floor that still moves. With
                # the floor at 0.35 of the commanded rate, one 50 Hz cycle is
                # well under a tenth of a degree, so aiming at the target rather
                # than at a tolerance band costs nothing in overshoot.
                # plan.expected_turn_s() models exactly this; change them together.
                mag = abs(prim.wz)
                taper = min(1.0, max(taper_floor(mag, self.lim.turn_min_wz), abs(remaining)
                                     / max(1e-6, math.radians(TAPER_DEG))))
                sign = 1.0 if remaining > 0 else -1.0
                a = min(1.0, max(0.0, elapsed / self.lim.ramp_s)) if self.lim.ramp_s > 0 else 1.0
                wz = sign * mag * taper * a
                return Output(0.0, 0.0, wz, RUNNING)
        elif prim.op == "stop":
            # "stop" has no duration: it means zero velocity, now. Without this,
            # `duration_s is None` fell through and the primitive never ended --
            # which only shows up on a plan whose LAST step is a stop.
            finished = True
            note = "stop"
        else:
            if prim.duration_s is not None and elapsed >= prim.duration_s:
                finished = True

        if finished:
            rec.actual_s = elapsed
            if self.i + 1 >= len(self.prims):
                return self._finish(t, note)
            self._advance(t)
            nxt = self.current.summary() if self.current else ""
            vx, wz = self._ramped(t, self.current)
            return Output(vx, 0.0, wz, RUNNING,
                          (note + " | next: " + nxt) if note else "next: " + nxt)

        vx, wz = self._ramped(t, prim)
        return Output(vx, 0.0, wz, RUNNING)

    # ---------------------------------------------------------------- endings
    def _finish(self, t, note=None):
        """Ramp to zero and hold it for settle_s before declaring DONE, so the
        process does not exit while the robot is still moving and let the deadman
        do the stopping."""
        if self._t_settle is None:
            self._t_settle = t
        if t - self._t_settle < self.lim.settle_s:
            return Output(0.0, 0.0, 0.0, RUNNING, note)
        self.state = DONE
        return Output(0.0, 0.0, 0.0, DONE, note or "plan complete")

    def _abort(self, reason, t):
        self.state = ABORTED
        self.abort_reason = reason
        if self.records and self.records[-1].aborted is None:
            self.records[-1].aborted = reason
            if self._t_prim is not None:
                self.records[-1].actual_s = max(0.0, t - self._t_prim)
        return Output(0.0, 0.0, 0.0, ABORTED, "ABORT: " + reason)

    def interrupt(self, reason, t=None):
        """An abort from outside the control loop (the operator's Ctrl-C). `t` is
        the executor's own clock; without it, the time of the last step() is used.
        Stage C found the caller passing 0.0 here, which recorded the interrupted
        step's actual_s as 0 (or negative) and made an interrupted walk report
        having walked nowhere."""
        if self.state in (DONE, ABORTED):
            return None
        if t is None:
            t = self._t_last if self._t_last is not None else (self._t_prim or 0.0)
        return self._abort(reason, t)

    def report(self):
        return {"executed": [r.to_dict() for r in self.records],
                "final_state": self.state,
                "abort_reason": self.abort_reason}
