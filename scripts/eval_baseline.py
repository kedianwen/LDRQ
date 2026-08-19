# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure a trained policy's velocity-tracking baseline table (Week04, PG-1).

Rolls a checkpoint out at a grid of *fixed* commanded forward speeds and
reports, per speed:

- **linear velocity tracking error** ``mean|v_x_body - cmd_x|`` -- this is the
  PG-1 acceptance number (target: <= 0.15 m/s over 0.5-1.0 m/s);
- **survival rate** -- fraction of episodes that did not end in a fall
  (time-outs don't count as failures);
- **energy** ``mean|tau * qdot|`` summed over joints, in watts;
- **action smoothness** ``mean|a_t - a_{t-1}|`` per action dim.

Commands are held at a point value by collapsing the command term's *ranges*
to ``(v, v)`` and zeroing ``rel_standing_envs``, rather than overwriting
``vel_command_b`` every step: the command manager resamples on its own
schedule and would otherwise fight a per-step overwrite (and zero out the
2% of envs it designates as standing envs).

``--friction`` / ``--push_vel`` turn this into a *stress* test for the
with-DR vs without-DR comparison: they move the evaluation off the nominal
conditions the Week03 policy was trained on.

Written before any Week04 change lands, deliberately: Week03 burned three
training rounds on gait verdicts made from tensorboard curves and sampled
video frames, both of which turned out to be unable to distinguish a healthy
gait from a broken one. See scripts/diagnose_gait.py for the per-foot
counterpart of this script.

Run from the project root:
    ~/IsaacLab/isaaclab.sh -p scripts/eval_baseline.py --headless --num_envs 64 \\
        --checkpoint logs/rsl_rl/r1_flat/<run_id>/model_1499.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

sys.path.insert(0, str(Path.home() / "IsaacLab" / "scripts" / "reinforcement_learning" / "rsl_rl"))
import cli_args  # noqa: E402  isort: skip


parser = argparse.ArgumentParser(description="Measure the PG-1 velocity-tracking baseline table for an R1 checkpoint.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to average over.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-R1-Play-v0", help="Name of the task.")
parser.add_argument(
    "--speeds", type=str, default="0.3,0.5,0.6,0.8,1.0", help="Comma-separated commanded forward speeds (m/s)."
)
parser.add_argument("--steps", type=int, default=500, help="Recorded control steps per speed (500 ~ 10s at 50Hz).")
parser.add_argument("--warmup", type=int, default=100, help="Steps discarded after each command switch (~2s).")
parser.add_argument(
    "--friction",
    type=float,
    default=None,
    help="Override ground friction (both static and dynamic). Stress-test knob; nominal training value is 1.0.",
)
parser.add_argument(
    "--push_vel",
    type=float,
    default=0.0,
    help="Lateral push magnitude in m/s, applied every --push_every steps. Stress-test knob.",
)
parser.add_argument("--push_every", type=int, default=250, help="Control steps between pushes (250 ~ 5s).")
parser.add_argument("--tag", type=str, default="", help="Suffix for the output filename, e.g. 'stress'.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import tasks.r1_flat  # noqa: E402, F401  -- registers the task
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def _pin_command(env, speed: float) -> None:
    """Hold every env's velocity command at (speed, 0, 0) for the rest of the run.

    Collapsing the *ranges* (rather than writing vel_command_b once) means the
    term's own periodic resampling keeps producing the same value, so the
    command stays pinned without this script having to fight it every step.
    """
    term = env.unwrapped.command_manager.get_term("base_velocity")
    term.cfg.ranges.lin_vel_x = (speed, speed)
    term.cfg.ranges.lin_vel_y = (0.0, 0.0)
    term.cfg.ranges.ang_vel_z = (0.0, 0.0)
    term.cfg.rel_standing_envs = 0.0
    term.is_standing_env[:] = False
    term.vel_command_b[:, 0] = speed
    term.vel_command_b[:, 1] = 0.0
    term.vel_command_b[:, 2] = 0.0


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    if args_cli.friction is not None:
        env_cfg.scene.terrain.physics_material.static_friction = args_cli.friction
        env_cfg.scene.terrain.physics_material.dynamic_friction = args_cli.friction
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join(str(_PROJECT_ROOT), "logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    dt = env.unwrapped.step_dt
    num_envs = env.unwrapped.num_envs
    run_label = f"{os.path.basename(os.path.dirname(resume_path))} / {os.path.basename(resume_path)}"
    speeds = [float(s) for s in args_cli.speeds.split(",")]

    rows = []
    obs, _ = env.get_observations()

    with torch.inference_mode():
        for speed in speeds:
            _pin_command(env, speed)

            err_sum = torch.zeros((), device=env.unwrapped.device)
            power_sum = torch.zeros((), device=env.unwrapped.device)
            smooth_sum = torch.zeros((), device=env.unwrapped.device)
            samples = 0
            falls = 0
            episodes = 0
            prev_actions = None

            for t in range(args_cli.warmup + args_cli.steps):
                actions = policy(obs)
                obs, _, dones, extras = env.step(actions)

                if args_cli.push_vel != 0.0 and t > 0 and t % args_cli.push_every == 0:
                    vel_w = robot.data.root_vel_w.clone()
                    # deterministic alternating sideways shove, so every env gets
                    # the same disturbance budget regardless of seed
                    vel_w[:, 1] += args_cli.push_vel if (t // args_cli.push_every) % 2 else -args_cli.push_vel
                    robot.write_root_velocity_to_sim(vel_w)

                if t < args_cli.warmup:
                    prev_actions = actions.clone()
                    continue

                # tracking error against the *commanded* forward speed, in the body frame
                v_x = robot.data.root_lin_vel_b[:, 0]
                err_sum += torch.abs(v_x - speed).mean()
                power_sum += torch.abs(robot.data.applied_torque * robot.data.joint_vel).sum(dim=1).mean()
                smooth_sum += torch.abs(actions - prev_actions).mean()
                samples += 1
                prev_actions = actions.clone()

                # a "fall" is any termination that isn't the episode timing out
                time_outs = extras.get("time_outs", torch.zeros_like(dones))
                falls += int((dones.bool() & ~time_outs.bool()).sum())
                episodes += int(dones.bool().sum())

            rows.append(
                {
                    "speed": speed,
                    "err": float(err_sum / max(samples, 1)),
                    "falls": falls,
                    "episodes": episodes,
                    "power": float(power_sum / max(samples, 1)),
                    "smooth": float(smooth_sum / max(samples, 1)),
                }
            )

    env.close()

    # One flushed write at the end: Isaac Sim's shutdown can kill the process
    # before Python flushes a block-buffered stdout (this silently ate
    # diagnose_gait.py's first report -- see scripts/README.md).
    out = []
    out.append("\n" + "=" * 78)
    out.append(f"Velocity-tracking baseline: {run_label}")
    conditions = f"friction={args_cli.friction if args_cli.friction is not None else 'nominal'}"
    if args_cli.push_vel:
        conditions += f", push={args_cli.push_vel} m/s every {args_cli.push_every * dt:.1f}s"
    out.append(f"{num_envs} envs, {args_cli.steps} steps ({args_cli.steps * dt:.1f}s) per speed, {conditions}")
    out.append("=" * 78)
    out.append("")
    out.append("| cmd v_x (m/s) | tracking err (m/s) | falls / episodes | energy (W) | action smoothness |")
    out.append("|---|---|---|---|---|")
    for r in rows:
        out.append(
            f"| {r['speed']:.1f} | {r['err']:.3f} | {r['falls']} / {r['episodes']} "
            f"| {r['power']:.1f} | {r['smooth']:.4f} |"
        )
    total_falls = sum(r["falls"] for r in rows)
    total_eps = sum(r["episodes"] for r in rows)
    out.append("")
    out.append(
        f"falls across all speeds: {total_falls} / {total_eps} episodes. Episode counts are"
        " coarse --\nenvs reset together, so a speed block shorter than an episode can log zero."
        " Read the\nfall count, not the ratio."
    )

    # PG-1 is stated over the 0.5-1.0 m/s band only
    pg1 = [r for r in rows if 0.5 - 1e-6 <= r["speed"] <= 1.0 + 1e-6]
    if pg1:
        worst = max(pg1, key=lambda r: r["err"])
        verdict = "PASS" if worst["err"] <= 0.15 else "FAIL"
        out.append("")
        out.append(
            f"PG-1 (tracking error <= 0.15 m/s over 0.5-1.0 m/s): {verdict} "
            f"-- worst is {worst['err']:.3f} m/s at {worst['speed']:.1f} m/s"
        )
    out.append("")
    out.append(
        "NOTE: tracking error alone does not describe the gait. A policy can track well\n"
        "while hopping on one leg -- run scripts/diagnose_gait.py alongside this."
    )

    report = "\n".join(out) + "\n"
    print(report, flush=True)
    name = f"baseline{'_' + args_cli.tag if args_cli.tag else ''}.md"
    report_path = os.path.join(os.path.dirname(resume_path), name)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[INFO] wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
