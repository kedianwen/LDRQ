# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record the simulation segment of the demo video: one R1 walking a mission_ctl script.

Same policy, same command generator and same default sequence as the stage A sweep
(scripts/sweep_gain_robustness.py), filmed by a camera that follows the robot. A
clip of the random commands play_r1.py samples would show "a robot walking"; this
one shows the exact sequence the real robot was asked to walk.

Run from the project root:
    ~/IsaacLab/isaaclab.sh -p scripts/record_demo_sim.py --headless \\
        --out outputs/demo/sim_sequence.mp4 [--kp_scale 1.0] [--script "..."]
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

parser = argparse.ArgumentParser(description="Record R1 walking a mission_ctl script in simulation.")
parser.add_argument("--out", type=str, required=True, help="Output .mp4 path.")
parser.add_argument("--kp_scale", type=float, default=1.0)
parser.add_argument("--script", type=str, default=None, help="mission_ctl script (default: the stage A sweep sequence).")
parser.add_argument("--policy", type=str, default=str(_PROJECT_ROOT / "models" / "week04_nohead" / "policy.pt"))
parser.add_argument("--settle_s", type=float, default=1.5)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-R1-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import tasks.r1_flat  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from gain_sweep_real import SWEEP_SCRIPT  # noqa: E402
from r1_mission import executor as EX  # noqa: E402
from r1_mission import node as MN  # noqa: E402
from r1_mission import plan as MP  # noqa: E402


def main():
    script = args_cli.script or SWEEP_SCRIPT
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.seed = 0
    env_cfg.episode_length_s = 120.0
    c = env_cfg.commands.base_velocity
    c.resampling_time_range = (1.0e6, 1.0e6)
    c.rel_standing_envs = 0.0
    c.debug_vis = False
    # follow the robot, three-quarter view from behind-left, at torso height
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (-3.6, 2.8, 1.6)
    env_cfg.viewer.lookat = (0.3, 0.0, 0.35)

    lim = MN.limits_from_cfg(MN.read_flat_yaml(str(_PROJECT_ROOT / "mission_ctl" / "config" / "mission.yaml")))
    envelope = MP.Envelope.from_bridge_yaml(
        _PROJECT_ROOT / "deploy" / "ros2_ws" / "src" / "r1_hw_bridge" / "config" / "bridge.yaml")
    prims, _, _ = MP.compile_plan(MP.parse_script(script), envelope, lim)
    ex = EX.Executor(prims, lim)

    dt_guess = env_cfg.sim.dt * env_cfg.decimation
    worst = sum(((p.expected_s or p.duration_s or 0.0) * lim.turn_timeout_factor) if p.closed_loop
                else (p.duration_s or 0.0) + lim.ramp_s for p in prims) + lim.settle_s
    max_steps = int((args_cli.settle_s + worst + 1.0) / dt_guess)

    tmp = tempfile.mkdtemp()
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=tmp, step_trigger=lambda s: s == 0,
                                   video_length=max_steps, disable_logger=True)
    env = RslRlVecEnvWrapper(env)
    uw = env.unwrapped
    robot = uw.scene["robot"]
    term = uw.command_manager.get_term("base_velocity")
    term.is_standing_env[:] = False
    dt = uw.step_dt
    if args_cli.kp_scale != 1.0:
        robot.write_joint_stiffness_to_sim(robot.data.default_joint_stiffness * args_cli.kp_scale)
        for act in robot.actuators.values():
            act.stiffness[:] = act.stiffness * args_cli.kp_scale

    policy = torch.jit.load(args_cli.policy, map_location=uw.device).eval()
    obs, _ = env.get_observations()
    settle = int(round(args_cli.settle_s / dt))
    tick = max(1, int(round(0.1 / dt)))           # mission_ctl publishes at 10 Hz
    yaw_prev, yaw_unw, vx, wz, tail = None, 0.0, 0.0, 0.0, None
    with torch.inference_mode():
        for k in range(max_steps):
            ke = k - settle
            if ke >= 0 and ke % tick == 0 and ex.state not in (EX.DONE, EX.ABORTED):
                o = ex.step(ke * dt, yaw=yaw_unw, bridge_state="RUNNING")
                vx, wz = o.vx, o.wz
            if ex.state in (EX.DONE, EX.ABORTED):
                tail = k if tail is None else tail
                if k - tail > int(1.0 / dt):          # one second of standing after, then stop
                    break
            term.vel_command_b[:, 0] = vx
            term.vel_command_b[:, 1] = 0.0
            term.vel_command_b[:, 2] = wz
            obs, _, dones, _ = env.step(policy(obs))
            if bool(dones.any()):
                print("[warn] the robot fell; the clip shows it", flush=True)
            q = robot.data.root_quat_w[0]
            yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
            if yaw_prev is not None:
                yaw_unw += (yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi
            yaw_prev = yaw
    rep = ex.report()
    env.close()                                        # RecordVideo writes the file on close
    clips = sorted(Path(tmp).glob("*.mp4"))
    if not clips:
        raise SystemExit("[fail] no video was written")
    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(clips[0]), out)
    turns = [r.get("achieved_deg") for r in rep["executed"] if r.get("op") == "turn"]
    print(f"[ok] wrote {out}  (mission {rep['final_state']}, turns {turns}, kp_scale {args_cli.kp_scale})", flush=True)
    os.rmdir(tmp) if not any(Path(tmp).iterdir()) else None


if __name__ == "__main__":
    main()
    simulation_app.close()
