# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quantify a trained policy's per-foot gait statistics (Week03 gait debugging).

Rolls out a checkpoint like ``play_r1.py`` does, but instead of rendering it
records each foot's contact/air state every control step and reports:

- air-time fraction, swing count, mean swing duration -- per foot, so a
  "left leg walks, right leg just taps" gait shows up as a number instead of
  a visual impression;
- each foot's dominant stepping frequency (FFT of its contact signal) and the
  left/right phase offset at that frequency. A proper walk is anti-phase
  (offset ~0.5 cycles); a hop/shuffle is in-phase (~0.0).

Written because the ``feet_air_time_symmetry`` reward rounds kept being
evaluated by eye -- see
~/kdw/experiment_record/Week03_reward_fix_symmetry_方案A失败复盘与v2方案.md.
Needs no GUI; run headless.

Run from the project root:
    ~/IsaacLab/isaaclab.sh -p scripts/diagnose_gait.py --headless --num_envs 32 \\
        --steps 600 \\
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


parser = argparse.ArgumentParser(description="Measure per-foot gait statistics for a trained R1 checkpoint.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to average over.")
parser.add_argument("--steps", type=int, default=600, help="Control steps to record (600 ~ 12s at 50Hz).")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-R1-Play-v0", help="Name of the task.")
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
import numpy as np
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import tasks.r1_flat  # noqa: E402, F401  -- registers the task
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

_FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]


def _swing_stats(air_signal: np.ndarray, dt: float) -> tuple[int, float, float]:
    """(swing count, mean swing duration, air fraction) for one foot in one env.

    ``air_signal`` is a boolean per-step trace (True = foot off the ground).
    Runs touching either end of the window are dropped from the duration stats
    since they're truncated, but still count toward the air fraction.
    """
    air_fraction = float(air_signal.mean())
    padded = np.concatenate([[False], air_signal, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    durations = (ends - starts) * dt
    # drop runs clipped by the recording window
    interior = (starts > 0) & (ends < len(air_signal))
    interior_durations = durations[interior]
    mean_duration = float(interior_durations.mean()) if interior_durations.size else 0.0
    return int(interior.sum()), mean_duration, air_fraction


def _dominant_frequency_and_phase(sig_a: np.ndarray, sig_b: np.ndarray, dt: float) -> tuple[float, float, float]:
    """Dominant stepping frequency of each foot, plus their phase offset in cycles.

    Phase offset is measured at foot A's dominant frequency. 0.5 = perfectly
    alternating (a real walk); 0.0 = both feet moving together (a hop).
    """
    a = sig_a.astype(np.float64) - sig_a.mean()
    b = sig_b.astype(np.float64) - sig_b.mean()
    n = len(a)
    freqs = np.fft.rfftfreq(n, d=dt)
    fft_a, fft_b = np.fft.rfft(a), np.fft.rfft(b)
    # ignore DC
    mag_a, mag_b = np.abs(fft_a), np.abs(fft_b)
    mag_a[0] = mag_b[0] = 0.0
    peak_a, peak_b = int(mag_a.argmax()), int(mag_b.argmax())
    if mag_a[peak_a] <= 0.0:
        return 0.0, 0.0, 0.0
    phase_diff = np.angle(fft_b[peak_a]) - np.angle(fft_a[peak_a])
    # wrap to [0, 1) cycles; 0.5 means anti-phase
    offset_cycles = float((phase_diff / (2 * np.pi)) % 1.0)
    return float(freqs[peak_a]), float(freqs[peak_b]), offset_cycles


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
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

    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
    foot_ids, foot_names = contact_sensor.find_bodies(_FEET, preserve_order=True)
    print(f"[INFO] Tracking feet {foot_names} (ids {foot_ids})")

    dt = env.unwrapped.step_dt
    num_envs = env.unwrapped.num_envs
    run_label = f"{os.path.basename(os.path.dirname(resume_path))} / {os.path.basename(resume_path)}"
    obs, _ = env.get_observations()

    # (steps, num_envs, 2) boolean: True when that foot is off the ground
    air_trace = np.zeros((args_cli.steps, num_envs, len(foot_ids)), dtype=bool)

    with torch.inference_mode():
        for t in range(args_cli.steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            # same contact criterion the reward functions use
            contact_time = contact_sensor.data.current_contact_time[:, foot_ids]
            air_trace[t] = (contact_time <= 0.0).cpu().numpy()

    env.close()

    # Build the whole report as a string and emit it in one flushed write:
    # Isaac Sim's shutdown can kill the process before Python flushes a
    # block-buffered stdout, which silently ate this report when redirected
    # to a file the first time round.
    out: list[str] = []
    out.append("\n" + "=" * 72)
    out.append(f"Gait diagnosis: {run_label}")
    out.append(f"{args_cli.steps} steps @ dt={dt:.4f}s ({args_cli.steps * dt:.1f}s), {num_envs} envs")
    out.append("=" * 72)

    per_foot = {name: {"count": [], "duration": [], "fraction": []} for name in foot_names}
    for env_idx in range(air_trace.shape[1]):
        for foot_idx, name in enumerate(foot_names):
            count, duration, fraction = _swing_stats(air_trace[:, env_idx, foot_idx], dt)
            per_foot[name]["count"].append(count)
            per_foot[name]["duration"].append(duration)
            per_foot[name]["fraction"].append(fraction)

    out.append(f"\n{'foot':<26}{'swings':>10}{'mean swing (s)':>18}{'air fraction':>16}")
    for name in foot_names:
        stats = per_foot[name]
        out.append(
            f"{name:<26}{np.mean(stats['count']):>10.1f}"
            f"{np.mean(stats['duration']):>18.3f}{np.mean(stats['fraction']):>16.3f}"
        )

    left, right = foot_names[0], foot_names[1]
    swing_ratio = np.mean(per_foot[right]["duration"]) / max(np.mean(per_foot[left]["duration"]), 1e-6)
    count_ratio = np.mean(per_foot[right]["count"]) / max(np.mean(per_foot[left]["count"]), 1e-6)
    out.append(f"\nright/left mean-swing-duration ratio: {swing_ratio:.3f}   (1.0 = symmetric)")
    out.append(f"right/left swing-count ratio:         {count_ratio:.3f}   (1.0 = symmetric)")
    out.append(
        "\nair fraction is the headline number: ~0.4-0.5 for a real walk (each foot "
        "airborne\nroughly the swing half of its cycle). A foot near 0.0 is a permanently "
        "planted\nsupport leg; a foot near 1.0 is a leg held up in the air."
    )

    freqs_l, freqs_r, offsets = [], [], []
    for env_idx in range(air_trace.shape[1]):
        f_l, f_r, offset = _dominant_frequency_and_phase(
            air_trace[:, env_idx, 0], air_trace[:, env_idx, 1], dt
        )
        if f_l > 0:
            freqs_l.append(f_l)
            freqs_r.append(f_r)
            offsets.append(offset)

    if freqs_l:
        # offsets are circular: average via unit vectors so 0.02 and 0.98 don't average to 0.5
        angles = np.array(offsets) * 2 * np.pi
        mean_offset = float(np.angle(np.mean(np.exp(1j * angles))) / (2 * np.pi) % 1.0)
        concentration = float(np.abs(np.mean(np.exp(1j * angles))))
        out.append(
            f"\ndominant stepping frequency: {left} {np.mean(freqs_l):.2f} Hz, {right} {np.mean(freqs_r):.2f} Hz"
        )
        out.append(
            f"left->right phase offset:    {mean_offset:.3f} cycles (0.5 = alternating walk, 0.0 = in-phase hop)"
        )
        out.append(f"  consistency across envs:   {concentration:.2f} (1.0 = every env agrees, 0.0 = no common phase)")
        out.append(
            "\nNOTE: read phase offset only together with the air fractions above. If one\n"
            "foot is planted and the other is held up, their contact traces are near-\n"
            "complements, so this reports ~0.5 ('alternating') for a gait that never steps."
        )

    report = "\n".join(out) + "\n"
    print(report, flush=True)
    report_path = os.path.join(os.path.dirname(resume_path), "gait_diagnosis.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[INFO] wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
