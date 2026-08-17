# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play/replay a trained checkpoint for Isaac-Velocity-Flat-R1-Play-v0 (Week03, step 4).

Project's own copy of IsaacLab's official
``scripts/reinforcement_learning/rsl_rl/play.py`` -- same reason as
``train_r1.py``: the official script only imports ``isaaclab_tasks``, so it
can't see ``tasks/r1_flat`` (an external package, see tasks/README.md).

In headless mode there's no GUI to watch, so pass --video to get a
recording and an automatic exit after --video_length steps; otherwise the
sim-step loop runs forever until killed (matches upstream play.py's
behavior, unchanged here).

Run from the project root, e.g. against Week03's first training run:
    ~/IsaacLab/isaaclab.sh -p scripts/play_r1.py \\
        --task=Isaac-Velocity-Flat-R1-Play-v0 --num_envs 16 --headless \\
        --video --video_length 400 \\
        --checkpoint logs/rsl_rl/r1_flat/2026-08-14_14-22-09_first_train/model_1499.pt
Omit --checkpoint to auto-resume the latest checkpoint of the latest run
under logs/rsl_rl/r1_flat/ instead.
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


parser = argparse.ArgumentParser(description="Play a trained checkpoint for Isaac-Velocity-Flat-R1-Play-v0.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video, then exit.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-R1-Play-v0", help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import tasks.r1_flat  # noqa: E402, F401  -- registers Isaac-Velocity-Flat-R1-Play-v0
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def main():
    """Play with RSL-RL agent."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # this project's runs live under <project_root>/logs, not IsaacLab's
    log_root_path = os.path.join(str(_PROJECT_ROOT), "logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        policy_nn = ppo_runner.alg.actor_critic

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    obs, _ = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
