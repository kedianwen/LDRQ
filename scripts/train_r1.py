# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train Isaac-Velocity-Flat-R1-v0 with rsl_rl PPO (Week03, FR-T4 first training).

This project's own copy of IsaacLab's official
``scripts/reinforcement_learning/rsl_rl/train.py``. Can't use the official
script unmodified: it only ``import isaaclab_tasks`` to trigger task
registration, which has no way to know about ``tasks/r1_flat`` -- an external
package living outside the IsaacLab install (see tasks/README.md for why it's
external). This copy adds that one import before the task registry is looked
up, and reuses IsaacLab's own ``cli_args`` module via a sys.path insert rather
than duplicating its CLI-argument logic.

Run from the project root:
    ~/IsaacLab/isaaclab.sh -p scripts/train_r1.py --task=Isaac-Velocity-Flat-R1-v0 \
        --headless --num_envs 4096 --max_iterations 1500

Checkpoint-resume (to stop mid-training, inspect gait with play_r1.py, then
continue toward the same original target instead of restarting from zero):
    # 1. train partway, e.g. to iteration 300, note the run's timestamp folder
    ~/IsaacLab/isaaclab.sh -p scripts/train_r1.py --task=Isaac-Velocity-Flat-R1-v0 \
        --headless --num_envs 4096 --max_iterations 300 --run_name my_run

    # 2. inspect with play_r1.py against that run's model_300.pt (see play_r1.py)

    # 3. resume to the *same original target* -- --max_iterations here is the
    #    absolute target iteration count, not "how many more to run". This
    #    differs from rsl_rl/IsaacLab's own upstream train.py, where --resume
    #    treats --max_iterations as an additional-iterations count on top of
    #    wherever the checkpoint left off (easy to misread as an absolute
    #    target and end up training past where you meant to stop).
    ~/IsaacLab/isaaclab.sh -p scripts/train_r1.py --task=Isaac-Velocity-Flat-R1-v0 \
        --headless --num_envs 4096 --max_iterations 1500 --run_name my_run \
        --resume --load_run <the_run_folder_from_step_1> --load_checkpoint model_300.pt
Resuming always writes a *new* timestamped log folder (weights/iteration
counter are loaded from the old one, but training continues into a fresh
directory) -- this matches upstream rsl_rl behavior, so a fully continued
training arc for one experiment ends up spread across more than one
logs/rsl_rl/r1_flat/<run_id>/ folder. Reuse the same --run_name across the
resume chain to keep them identifiable as one arc when browsing logs/.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

# local imports (IsaacLab's own cli_args.py, reused rather than duplicated)
sys.path.insert(0, str(Path.home() / "IsaacLab" / "scripts" / "reinforcement_learning" / "rsl_rl"))
import cli_args  # noqa: E402  isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train Isaac-Velocity-Flat-R1-v0 with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-R1-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import tasks.r1_flat  # noqa: E402, F401  -- registers Isaac-Velocity-Flat-R1-v0
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # logs live in this project's own logs/ dir, not IsaacLab's
    log_root_path = os.path.join(str(_PROJECT_ROOT), "logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO] actor obs dim: {env.num_obs}, critic obs dim: {env.num_privileged_obs}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    if agent_cfg.resume:
        # rsl_rl's OnPolicyRunner.learn(num_learning_iterations=N) always runs N
        # *more* iterations from wherever runner.load() left current_learning_iteration
        # -- treating --max_iterations as an additional count, not an absolute
        # target. That's an easy footgun for pause-inspect-resume workflows
        # (see module docstring), so re-interpret --max_iterations as the
        # absolute target here and compute the remaining count ourselves.
        remaining = max(agent_cfg.max_iterations - runner.current_learning_iteration, 0)
        print(
            f"[INFO] Resuming from iteration {runner.current_learning_iteration}: running "
            f"{remaining} more to reach --max_iterations={agent_cfg.max_iterations}"
        )
        num_learning_iterations = remaining
    else:
        num_learning_iterations = agent_cfg.max_iterations

    runner.learn(num_learning_iterations=num_learning_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
