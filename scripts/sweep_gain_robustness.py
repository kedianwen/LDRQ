# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulated half of the kp_scale stability-domain sweep (stage A: FR-R2 / FR-R4).

The real half is deploy/tools/probe_cpp/gain_sweep_real.py. The two halves only
measure a sim2real gap if they ask the same question the same way, so this
script deliberately borrows rather than re-implements:

* the command sequence -- ``SWEEP_SCRIPT`` is imported from gain_sweep_real.py,
  the exact string the robot walked;
* the command generator -- every env drives its own ``mission_ctl`` Executor
  (pure Python), ticked at mission_ctl's 10 Hz with that env's own IMU-style
  yaw, so ramps and the closed-loop turn taper are the robot's, not a retyped
  timetable;
* the gain knob -- kp_scale multiplies EVERY actuator's stiffness, kd untouched,
  which is what the bridge's kp_scale launch argument does;
* the policy -- ``models/week04_nohead/policy.pt`` (the deployed run's export),
  checked at start-up against the parity_fixture.bin next to it (the fixture the robot's TensorRT
  engine matched to 1.3e-5), so "the sim" is provably the network on the robot;
* the scoring -- over the COMMANDED window only (first to last non-zero
  command), with the standing tilt before the first command reported apart.

All gains run in ONE rollout: env i belongs to point i // envs_per_point, and
stiffness is written per env at runtime. Nominal conditions (the PLAY config:
no DR, no pushes, no action delay).

Stability definition -- fixed before this script was first run, 2026-09-29, and
the same as the real side's, applied per env:
  * no fall (the task's own terminations: pelvis below 0.51 m or tilt > 0.7 rad)
    -- the sim's counterpart of the bridge going DEGRADED;
  * torso tilt <= 20 deg over the commanded window;
  * leg (hip+knee+ankle) tracking rms <= 1.5x the kp=1.0 point's median;
  * computed (pre-clip) actuator torque below each joint's effort limit.
A point is STABLE when at least 95% of its envs pass. (The real side's rule is
"every recording passes" because it has one or two recordings per gain; with 64
envs per gain a pass RATE is the meaningful number, and it is reported as the
sim curve.)

Run from the project root (writes outputs/gain_sweep/<date>/results.json, which
``gain_sweep_real.py --collect <dir> --sim <results.json>`` reads):
    ~/IsaacLab/isaaclab.sh -p scripts/sweep_gain_robustness.py --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "mission_ctl"))
sys.path.insert(0, str(_PROJECT_ROOT / "deploy" / "tools" / "probe_cpp"))

_RUN = "2026-08-19_11-03-32_week04_nohead"

parser = argparse.ArgumentParser(description="Simulated kp_scale stability-domain sweep.")
parser.add_argument("--points", type=str, default=",".join(f"{0.5 + 0.1 * i:.1f}" for i in range(16)),
                    help="Comma-separated kp_scale values (default 0.5..2.0 step 0.1). Must include 1.0.")
parser.add_argument("--envs_per_point", type=int, default=64)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-R1-Play-v0")
# models/week04_nohead/ is the copy tracked in git (a fresh clone has no logs/);
# both are byte-identical to what was installed on the robot.
_MODELS = _PROJECT_ROOT / "models" / "week04_nohead"
parser.add_argument("--policy", type=str, default=str(_MODELS / "policy.pt"))
parser.add_argument("--fixture", type=str, default=str(_MODELS / "parity_fixture.bin"))
parser.add_argument("--settle_s", type=float, default=3.0, help="Zero command before the sequence starts.")
parser.add_argument("--stand_s", type=float, default=1.7,
                    help="Tail of the settle window scored as standing tilt (the real recordings had 1.7 s).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", type=str, default=None, help="results.json path (default outputs/gain_sweep/<date>/).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import datetime  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import struct  # noqa: E402
import subprocess  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import tasks.r1_flat  # noqa: E402, F401  -- registers the task
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from gain_sweep_real import SWEEP_SCRIPT  # noqa: E402  -- the string the robot walked
from r1_mission import executor as EX  # noqa: E402
from r1_mission import node as MN  # noqa: E402  (rclpy is imported lazily there)
from r1_mission import plan as MP  # noqa: E402

TILT_LIMIT_DEG = 20.0
TRACKING_FACTOR = 1.5
STABLE_PASS_RATE = 0.95
MISSION_HZ = 10.0

_LOG = []


def say(msg=""):
    """Print flushed AND keep a copy: Isaac Sim's shutdown can eat a buffered stdout."""
    print(msg, flush=True)
    _LOG.append(msg)


def check_policy_is_deployed(policy, fixture_path, device):
    with open(fixture_path, "rb") as fh:
        if fh.read(4) != b"R1FX":
            raise SystemExit(f"[fail] {fixture_path}: bad fixture magic")
        _, n, ind, outd = struct.unpack("<IIII", fh.read(16))
        ins = torch.frombuffer(bytearray(fh.read(n * ind * 4)), dtype=torch.float32).reshape(n, ind)
        ref = torch.frombuffer(bytearray(fh.read(n * outd * 4)), dtype=torch.float32).reshape(n, outd)
    with torch.inference_mode():
        got = policy(ins.to(device)).float().cpu()
    err = float((got - ref).abs().max())
    if err > 1e-4:
        raise SystemExit(f"[fail] the policy does not reproduce the deployed parity fixture (max err {err:.3g})")
    return n, err


def pct(t, q):
    """Per-column quantile of a [T, N] tensor, ignoring NaNs (outside-window samples)."""
    return torch.nanquantile(t, q, dim=0)


def main():
    points = [round(float(x), 3) for x in args_cli.points.split(",")]
    if 1.0 not in points:
        raise SystemExit("[fail] --points must include 1.0: it is the tracking baseline")
    E = args_cli.envs_per_point
    N = len(points) * E

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=N,
                            use_fabric=not args_cli.disable_fabric)
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = 120.0          # the sequence is ~30 s; the task's 20 s would reset mid-turn
    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.resampling_time_range = (1.0e6, 1.0e6)   # this script owns the command
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.heading_command = False
    cmd_cfg.debug_vis = False
    env_cfg.scene.env_spacing = 4.0           # 1.5 m out-and-back plus spin-up room

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    uw = env.unwrapped
    dev = uw.device
    robot = uw.scene["robot"]
    term = uw.command_manager.get_term("base_velocity")
    term.is_standing_env[:] = False
    dt = uw.step_dt

    policy = torch.jit.load(args_cli.policy, map_location=dev)
    policy.eval()
    nfix, fixerr = check_policy_is_deployed(policy, args_cli.fixture, dev)
    say(f"[ok] policy reproduces the deployed parity fixture: {nfix} vectors, max err {fixerr:.2e}")

    # ---- the knob: kp_scale on every actuator's stiffness, per env ----
    kp_env = torch.tensor([points[i // E] for i in range(N)], device=dev)
    base_stiff = robot.data.default_joint_stiffness.clone()
    robot.write_joint_stiffness_to_sim(base_stiff * kp_env[:, None])
    for act in robot.actuators.values():
        # write_joint_stiffness_to_sim only reaches PhysX; the actuator model's own
        # copy is what computed_torque is built from, so it has to follow.
        act.stiffness[:] = act.stiffness * kp_env[:, None]
    eff = torch.zeros(robot.num_joints, device=dev)
    leg_ids = []
    for name, act in robot.actuators.items():
        ids = act.joint_indices
        if isinstance(ids, slice):
            ids = list(range(robot.num_joints))
        elif torch.is_tensor(ids):
            ids = ids.tolist()
        else:
            ids = [int(i) for i in ids]
        eff[ids] = act.effort_limit[0].to(dev) if torch.is_tensor(act.effort_limit) else float(act.effort_limit)
        if name in ("legs", "ankles"):
            leg_ids += ids
    if not leg_ids:
        raise SystemExit("[fail] no 'legs'/'ankles' actuator groups -- the tracking criterion needs them")
    leg_ids = torch.tensor(sorted(leg_ids), device=dev)

    # ---- the command generator: one mission_ctl Executor per env ----
    envelope = MP.Envelope.from_bridge_yaml(
        _PROJECT_ROOT / "deploy" / "ros2_ws" / "src" / "r1_hw_bridge" / "config" / "bridge.yaml")
    lim = MN.limits_from_cfg(MN.read_flat_yaml(str(_PROJECT_ROOT / "mission_ctl" / "config" / "mission.yaml")))
    prims, plan_s, _ = MP.compile_plan(MP.parse_script(SWEEP_SCRIPT), envelope, lim)
    execs = [EX.Executor(prims, lim) for _ in range(N)]
    say(f"[ok] {len(points)} gains x {E} envs = {N}; sequence '{SWEEP_SCRIPT}' (~{plan_s:.1f} s as executed)")

    settle_steps = int(round(args_cli.settle_s / dt))
    stand_steps = int(round(args_cli.stand_s / dt))
    tick_every = max(1, int(round(1.0 / (MISSION_HZ * dt))))
    worst = sum(((p.expected_s or p.duration_s or 0.0) * lim.turn_timeout_factor) if p.closed_loop
                else (p.duration_s or 0.0) + lim.ramp_s for p in prims) + lim.settle_s + 2.0
    T = settle_steps + int(math.ceil(worst / dt))

    nan = float("nan")
    tilt_log = torch.full((T, N), nan, device=dev)
    legsq_log = torch.full((T, N), nan, device=dev)
    tau_log = torch.full((T, N), nan, device=dev)
    flip_log = torch.zeros((T, N), device=dev)
    moving = torch.zeros((T, N), dtype=torch.bool, device=dev)
    fell_at = torch.full((N,), -1, dtype=torch.long, device=dev)

    vx = torch.zeros(N, device=dev)
    wz = torch.zeros(N, device=dev)
    yaw_prev = None
    yaw_unw = torch.zeros(N, device=dev)
    a1 = a2 = None
    obs, _ = env.get_observations()
    last_step = T - 1

    with torch.inference_mode():
        for k in range(T):
            ke = k - settle_steps                       # executor-phase step index
            if ke >= 0 and ke % tick_every == 0:
                t = ke * dt
                yaws = yaw_unw.tolist()
                fell = (fell_at >= 0).tolist()
                ovx, owz = [0.0] * N, [0.0] * N
                active = False
                for i, e in enumerate(execs):
                    if e.state in (EX.DONE, EX.ABORTED):
                        continue
                    o = e.step(t, yaw=yaws[i], bridge_state="DEGRADED" if fell[i] else "RUNNING")
                    ovx[i], owz[i] = o.vx, o.wz
                    active = True
                vx = torch.tensor(ovx, device=dev)
                wz = torch.tensor(owz, device=dev)
                if not active:
                    last_step = k
                    break
            term.vel_command_b[:, 0] = vx
            term.vel_command_b[:, 1] = 0.0
            term.vel_command_b[:, 2] = wz

            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

            # a done inside a 120 s episode is a fall, never a time-out
            new_fall = dones.bool() & (fell_at < 0)
            fell_at[new_fall] = k

            g = robot.data.projected_gravity_b
            tilt_log[k] = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2] / g.norm(dim=1), -1.0, 1.0)))
            err = robot.data.joint_pos_target[:, leg_ids] - robot.data.joint_pos[:, leg_ids]
            legsq_log[k] = (err * err).mean(dim=1)
            tau_log[k] = (robot.data.computed_torque.abs() / eff).max(dim=1).values
            moving[k] = (vx.abs() > 1e-3) | (wz.abs() > 1e-3)
            if a2 is not None:
                flip_log[k] = (((a1 - a2) * (actions - a1)) < 0).float().mean(dim=1)
            a2, a1 = a1, actions.clone()

            q = robot.data.root_quat_w           # (w, x, y, z)
            yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            if yaw_prev is not None:
                d = yaw - yaw_prev
                yaw_unw += torch.remainder(d + math.pi, 2 * math.pi) - math.pi
            yaw_prev = yaw

    T_used = last_step + 1
    idx = torch.arange(T_used, device=dev)[:, None]
    mv = moving[:T_used]
    any_mv = mv.any(dim=0)
    first = torch.where(any_mv, mv.float().argmax(dim=0), torch.zeros_like(fell_at))
    last = torch.where(any_mv, T_used - 1 - mv.flip(0).float().argmax(dim=0), torch.zeros_like(fell_at))
    in_win = (idx >= first[None]) & (idx <= last[None])
    in_stand = (idx >= settle_steps - stand_steps) & (idx < settle_steps)

    def win(t):
        return torch.where(in_win, t[:T_used], torch.full_like(t[:T_used], nan))

    # max over the window: fill outside-window samples low rather than NaN (amax propagates NaN)
    tilt_max = torch.where(in_win, tilt_log[:T_used], torch.full_like(tilt_log[:T_used], -1.0)).amax(dim=0)
    tilt_p99 = pct(win(tilt_log), 0.99)
    leg_rms = torch.sqrt(torch.nanmean(win(legsq_log), dim=0))
    tau_max = torch.where(in_win, tau_log[:T_used], torch.full_like(tau_log[:T_used], -1.0)).amax(dim=0)
    tau_p99 = pct(win(tau_log), 0.99)
    stand_tilt = torch.nanmedian(torch.where(in_stand, tilt_log[:T_used], torch.full_like(tilt_log[:T_used], nan)),
                                 dim=0).values
    n_win = in_win.float().sum(dim=0).clamp(min=1)
    chatter = (torch.where(in_win, flip_log[:T_used], torch.zeros_like(flip_log[:T_used])).sum(dim=0) / n_win) \
        * (1.0 / dt) / 2.0
    fell = fell_at >= 0

    base_mask = kp_env == 1.0
    baseline = float(torch.median(leg_rms[base_mask & ~fell])) if bool((base_mask & ~fell).any()) \
        else float(torch.median(leg_rms[base_mask]))

    reasons = {
        "fell": fell,
        "tilt": tilt_max > TILT_LIMIT_DEG,
        "tracking": leg_rms > TRACKING_FACTOR * baseline,
        "torque": tau_max >= 1.0,
    }
    ok = ~(reasons["fell"] | reasons["tilt"] | reasons["tracking"] | reasons["torque"])

    reports = [e.report() for e in execs]
    out_pts = []
    for pi, kp in enumerate(points):
        s = slice(pi * E, (pi + 1) * E)
        turn_s = [r.get("actual_s") for rep in reports[s] for r in rep["executed"] if r.get("op") == "turn"]
        turn_s = [x for x in turn_s if x is not None]
        states = [rep["final_state"] for rep in reports[s]]
        pr = float(ok[s].float().mean())
        # when the falls happened, relative to the first command: negative = while
        # still standing in the settle window, before anything was commanded
        fall_t = ((fell_at[s][fell[s]] - settle_steps).float() * dt).tolist()
        out_pts.append({
            "kp_scale": kp,
            "stable": pr >= STABLE_PASS_RATE,
            "pass_rate": pr,
            "n_envs": E,
            "fails": {k: int(v[s].sum()) for k, v in reasons.items()},
            "mission_done": states.count("DONE"),
            "stand_tilt_deg": float(torch.nanmedian(stand_tilt[s])),
            "tilt_max_deg": float(torch.median(tilt_max[s])),
            "tilt_max_deg_worst": float(tilt_max[s].max()),
            "tilt_p99_deg": float(torch.nanmedian(tilt_p99[s])),
            "leg_track_rms": float(torch.nanmedian(leg_rms[s])),
            "leg_track_ratio": float(torch.nanmedian(leg_rms[s])) / baseline,
            "tau_frac_max": float(torch.median(tau_max[s])),
            "tau_frac_max_worst": float(tau_max[s].max()),
            "tau_frac_p99": float(torch.nanmedian(tau_p99[s])),
            "chatter_hz": float(torch.median(chatter[s])),
            "turn_s_median": (sorted(turn_s)[len(turn_s) // 2] if turn_s else None),
            "fall_time_s": ({"median": sorted(fall_t)[len(fall_t) // 2], "min": min(fall_t), "max": max(fall_t)}
                            if fall_t else None),
        })

    stable = [p["kp_scale"] for p in out_pts if p["stable"]]
    try:
        sha = subprocess.check_output(["git", "-C", str(_PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
                                      text=True).strip()
    except Exception:
        sha = "unknown"
    result = {
        "meta": {
            "what": "simulated kp_scale stability-domain sweep (stage A, FR-R2/FR-R4)",
            "date": datetime.date.today().isoformat(),
            "git": sha,
            "run": _RUN,
            "policy": os.path.relpath(args_cli.policy, _PROJECT_ROOT),
            "parity_fixture_max_err": fixerr,
            "task": args_cli.task,
            "conditions": "PLAY config: nominal friction/mass/gains, no pushes, no action delay",
            "sequence": SWEEP_SCRIPT,
            "mission_hz": MISSION_HZ,
            "envs_per_point": E,
            "settle_s": args_cli.settle_s,
            "stand_s": args_cli.stand_s,
            "seed": args_cli.seed,
            "definition": {
                "per_env": f"no fall; tilt <= {TILT_LIMIT_DEG} deg; leg tracking rms <= {TRACKING_FACTOR}x "
                           "the kp=1.0 median; computed torque below effort limit -- over the commanded window",
                "per_point": f"stable if pass rate >= {STABLE_PASS_RATE}",
            },
            "tracking_baseline_rms": baseline,
        },
        "points": out_pts,
        "stable": stable,
    }

    out = Path(args_cli.out) if args_cli.out else \
        _PROJECT_ROOT / "outputs" / "gain_sweep" / datetime.date.today().isoformat() / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    say("")
    say("=== simulated kp_scale sweep (commanded window; medians over envs) ===")
    say(f"{'kp':>5} {'pass':>5} {'stand':>6} {'tiltmax':>7} {'p99':>5} {'leg rms':>8} {'x base':>6} "
        f"{'tau max':>7} {'tau p99':>7} {'chatHz':>6} {'turn s':>6}  fails(fell/tilt/track/torque)")
    for p in out_pts:
        f = p["fails"]
        say(f"{p['kp_scale']:5.2f} {100 * p['pass_rate']:4.0f}% {p['stand_tilt_deg']:6.1f} {p['tilt_max_deg']:7.1f} "
            f"{p['tilt_p99_deg']:5.1f} {p['leg_track_rms']:8.4f} {p['leg_track_ratio']:6.2f} "
            f"{100 * p['tau_frac_max']:6.0f}% {100 * p['tau_frac_p99']:6.0f}% {p['chatter_hz']:6.2f} "
            f"{(p['turn_s_median'] or float('nan')):6.1f}  {f['fell']}/{f['tilt']}/{f['tracking']}/{f['torque']}"
            f"{'' if p['stable'] else '   <- not stable'}")
        if p["fall_time_s"]:
            ft = p["fall_time_s"]
            say(f"{'':>5}   falls at t = {ft['min']:+.1f} .. {ft['max']:+.1f} s from the first command "
                f"({'while STANDING, before any command' if ft['max'] < 0 else 'during the sequence'})")
    say("")
    if stable:
        lo, hi = min(stable), max(stable)
        say(f"sim stable domain: {lo:.2f} .. {hi:.2f} (width {hi - lo:.2f}); "
            f"lower edge {'BOUNDED' if lo > min(points) else 'NOT bounded'}, "
            f"upper edge {'BOUNDED' if hi < max(points) else 'NOT bounded'}")
        gaps = [k for k in points if lo < k < hi and k not in stable]
        if gaps:
            say(f"WARN  unstable points inside the interval: {gaps}")
    else:
        say("sim stable domain: empty")
    say(f"tracking baseline (kp=1.0 median leg rms): {baseline:.4f} rad")
    say(f"[ok] wrote {out}")
    (out.parent / "sweep_log.txt").write_text("\n".join(_LOG) + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
