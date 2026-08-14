#!/usr/bin/env python3
"""Random-action smoke test for the R1 flat-velocity task (Week02 exit criterion).

This is our project's equivalent of Isaac Lab's own
``scripts/environments/random_agent.py``. We can't use that script directly:
it only imports ``isaaclab_tasks`` (the built-in task package), so it has no
way to know about ``tasks/r1_flat`` living outside of Isaac Lab. This script
is that same ~30 lines with one addition: importing our task package first
so its ``gym.register()`` call actually runs.

Run from the project root::

    ~/IsaacLab/isaaclab.sh -p scripts/random_agent_r1.py --task=Isaac-Velocity-Flat-R1-v0 --num_envs 16
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Random agent smoke test for the R1 flat-velocity task.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-R1-v0", help="Name of the task.")
parser.add_argument("--steps", type=int, default=200, help="Number of steps to run before exiting.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----- after the app is up -----
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tasks.r1_flat  # noqa: E402, F401  (registers Isaac-Velocity-Flat-R1-v0)


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    env.reset()

    step = 0
    while simulation_app.is_running() and step < args_cli.steps:
        with torch.inference_mode():
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            env.step(actions)
        step += 1

    print(f"[INFO]: Completed {step} steps without crashing.")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
