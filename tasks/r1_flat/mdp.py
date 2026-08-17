# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R1-specific reward terms not covered by ``isaaclab_tasks``' shared locomotion mdp.

Week03 gait debugging. Full history in ~/kdw/experiment_record/, short version:

``mdp.feet_air_time_positive_biped`` is *maximized* by standing on one leg --
with one foot planted and one held up, ``single_stance`` is permanently true
and both feet's ``in_mode_time`` grow without bound, so the term sits clamped
at its ``threshold`` maximum forever. Measured on two separate trained
policies (``scripts/diagnose_gait.py``): left foot airborne 6% of the time,
right foot 95%. Two earlier rounds of left/right symmetry penalties failed to
fix this because they attacked a symptom, not that term.

The functions here replace it with a touchdown-settled swing reward that can't
be farmed by holding a leg up, plus a direct penalty on doing so.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_swing_touchdown(
    env: "ManagerBasedRLEnv",
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    min_swing_time: float = 0.15,
    threshold: float = 0.45,
) -> torch.Tensor:
    """Reward each *completed* swing, settled at the moment the foot touches down.

    Credit per touchdown is the swing's duration, floored and capped:

    - a foot that never lands is never paid -- kills the "hold one leg up
      forever" exploit that ``feet_air_time_positive_biped`` maximizes;
    - a swing shorter than ``min_swing_time`` pays zero -- kills the "tap the
      ground rapidly" exploit. (Note this is why credit isn't simply
      ``clamp(last_air, max=threshold)``: paying per touchdown proportionally
      to swing time makes rapid tapping *more* profitable than real stepping,
      since taps land far more often.)
    - credit saturates at ``threshold`` -- no bonus for dangling a leg longer.

    Because it only pays on touchdown (~2 frames per gait cycle rather than
    every frame), this term needs a much larger weight than the per-frame
    term it replaces to carry comparable influence.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    credit = torch.clamp(torch.clamp(last_air_time, max=threshold) - min_swing_time, min=0.0)
    reward = torch.sum(credit * first_contact, dim=1)
    # no reward when the robot isn't asked to move
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_excess_l1(
    env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg, max_air_time: float = 0.6
) -> torch.Tensor:
    """Penalize a foot staying airborne longer than ``max_air_time``.

    Fires every frame the foot is still up, so unlike a touchdown-gated or
    ``last_air_time``-based signal it doesn't have to wait for a landing that
    a held-up leg never makes. A normal swing (~0.3s) never reaches the
    threshold, so this is inert for healthy gaits.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    return torch.sum(torch.clamp(current_air_time - max_air_time, min=0.0), dim=1)
