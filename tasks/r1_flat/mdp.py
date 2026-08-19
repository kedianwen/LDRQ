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

Week04 adds ``command_range_curriculum`` -- Isaac Lab 2.1 ships no built-in
curriculum over command ranges (``envs/mdp/curriculums.py`` only has
``modify_reward_weight``).
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


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


class DelayedJointPositionAction(JointPositionAction):
    """Joint position action whose setpoint lags by a random number of control steps.

    This is Week04's control-delay randomization (FR-T5). It is *not* the
    obvious implementation -- Isaac Lab ships ``DelayedPDActuatorCfg`` for
    exactly this -- and the reason is measured, not stylistic:

    ``DelayedPDActuatorCfg`` is an *explicit* actuator, so swapping R1's legs
    onto it moves the PD from PhysX's implicit joint-drive solver into Python.
    Standing-tested with ``scripts/inspect_r1.py``, R1 collapsed in 1.5s (roll
    -10.6 deg by t=0.5s while pitch was still 1.3 deg, i.e. the *ankles* went
    first), and it collapsed identically with the delay set to zero, so the
    actuator model was at fault and not the lag. That matches the arithmetic:
    explicit damping needs roughly ``dt < 2J/d``, and the ankle joints run
    ``d=150`` against an inertia of order 0.01 kg m^2 (armature included),
    giving a limit near 1e-4 s against the 2e-3 s this env runs at. R1's leg
    gains only ever stood up because PhysX integrates them implicitly (see the
    Week02 diagnosis in assets/r1/r1.py), and an explicit model spends exactly
    that margin.

    Delaying the *action* instead reproduces the thing that actually needs
    modeling for sim2real -- the lag between an observation being taken and
    the resulting setpoint reaching the joints -- while leaving the PD where
    it works. It's also the more faithful unit: lags are counted in control
    steps (20ms at this env's 50Hz), which is the period the deployed policy
    loop will actually run at, rather than in 2ms physics substeps.
    """

    cfg: DelayedJointPositionActionCfg

    def __init__(self, cfg: DelayedJointPositionActionCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._delay_buffer = DelayBuffer(cfg.max_delay, self.num_envs, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        # called once per env (control) step, so the buffer's lag unit is control steps
        self._processed_actions = self._delay_buffer.compute(self._processed_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
            num_envs = self.num_envs
        else:
            num_envs = len(env_ids)
        # a fresh lag per episode, so a policy can't learn one fixed latency
        lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self.device,
        )
        self._delay_buffer.set_time_lag(lags, env_ids)
        self._delay_buffer.reset(env_ids)


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`DelayedJointPositionAction`."""

    class_type: type = DelayedJointPositionAction

    min_delay: int = 0
    """Minimum setpoint lag, in control steps (20ms each at this env's 50Hz)."""

    max_delay: int = 1
    """Maximum setpoint lag, in control steps. Also sizes the delay buffer."""


class body_material_friction(ManagerTermBase):
    """Per-env mean static/dynamic friction of the *feet's* collision shapes.

    A *privileged* observation for the critic only: the real R1 has no way to
    measure the ground it is standing on, but the critic may use it, and once
    Week04's ``randomize_rigid_body_material`` event makes friction vary
    per env, an unobserved friction is otherwise pure reward noise the value
    function has to eat.

    Averaging over *all* the robot's shapes instead of the feet's would be a
    one-liner, but it is a much weaker signal than it looks: friction is drawn
    per shape, so a mean over R1's 25 shapes has about a fifth of the spread of
    the feet's own value (measured: std 0.044 against the feet's ~0.17) and is
    only weakly correlated with the friction that actually decides whether a
    foot slips. Hence the shape-index mapping below.

    Friction is randomized at ``startup`` and never changes afterwards, so the
    lookup (a CPU-side PhysX query, far too slow to run every step) is cached.

    Caching on first use is *not* enough, and this bit was caught by
    measurement rather than reasoning: ``ObservationManager`` calls every term
    once while preparing terms, to discover its output shape, and that happens
    before ``load_managers`` applies the startup events. A cache filled on
    first call therefore pins the USD's default friction -- 1.000 for all
    envs, verified -- and the observation is silently constant. The cache is
    instead dropped once, on the first reset, which is after startup
    randomization; from then on it is never invalidated again, so the PhysX
    query still runs exactly once per training run.

    R1 has 27 bodies but 25 collision shapes, so a shape cannot be indexed by
    its body id. The per-body shape counts are recovered the same way Isaac
    Lab's own ``randomize_rigid_body_material`` does it (a per-link rigid-body
    view; that function calls it "a workaround since the Articulation does not
    provide a direct way"). If the counts ever fail to add up -- e.g. after an
    Isaac Lab upgrade changes that internal -- this falls back to the mean over
    all shapes rather than raising, since a degraded critic observation should
    not take a training run down.
    """

    def __init__(self, cfg: "ObservationTermCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._cached: torch.Tensor | None = None
        self._dropped_startup_cache = False

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # exactly once, on the first reset: discard whatever the shape-probe
        # call cached before the startup randomization ran
        if not self._dropped_startup_cache:
            self._cached = None
            self._dropped_startup_cache = True

    @staticmethod
    def _shape_ids_for_bodies(asset, body_ids: Sequence[int]) -> list[int] | None:
        """Collision-shape indices belonging to ``body_ids``, or None if unmappable."""
        # an unresolved SceneEntityCfg leaves body_ids as slice(None) -- means "all"
        if isinstance(body_ids, slice):
            return None
        counts = []
        for link_path in asset.root_physx_view.link_paths[0]:
            counts.append(asset._physics_sim_view.create_rigid_body_view(link_path).max_shapes)
        if sum(counts) != asset.root_physx_view.max_shapes:
            return None
        starts = [sum(counts[:i]) for i in range(len(counts))]
        return [s for body in body_ids for s in range(starts[body], starts[body] + counts[body])]

    def __call__(
        self,
        env: "ManagerBasedEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    ) -> torch.Tensor:
        if self._cached is None:
            asset = env.scene[asset_cfg.name]
            # (num_envs, num_shapes, 3): static friction, dynamic friction, restitution
            materials = asset.root_physx_view.get_material_properties().to(env.device)
            shape_ids = self._shape_ids_for_bodies(asset, asset_cfg.body_ids)
            if shape_ids:
                materials = materials[:, shape_ids]
            self._cached = materials[..., :2].mean(dim=1)
        return self._cached


def command_range_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids,  # noqa: ARG001  -- required by the curriculum-term signature, unused (ranges are global)
    command_name: str,
    initial_ranges: dict[str, tuple[float, float]],
    final_ranges: dict[str, tuple[float, float]],
    start_step: int,
    end_step: int,
) -> float:
    """Linearly widen a velocity command term's sampling ranges over training.

    Week04 has to train 0-1.0 m/s (PG-1 is stated over 0.5-1.0 m/s) and turning,
    where Week03 only ever trained straight-line 0-0.5 m/s. Opening the full
    range from step 0 throws away the one thing that already works; this ramps
    ``initial_ranges`` -> ``final_ranges`` between two values of
    ``env.common_step_counter``, which counts *control* steps, so with rsl_rl's
    24 steps per iteration, iteration N is step 24*N.

    Ranges live on the command term's cfg and are read at every resample, so
    mutating them here is enough -- there is nothing per-env to update, which is
    why ``env_ids`` is ignored. Returns the ramp fraction, which the curriculum
    manager logs as ``Curriculum/<term name>``.
    """
    span = max(end_step - start_step, 1)
    alpha = min(max((env.common_step_counter - start_step) / span, 0.0), 1.0)
    ranges = env.command_manager.get_term(command_name).cfg.ranges
    for key, (lo_final, hi_final) in final_ranges.items():
        lo_init, hi_init = initial_ranges[key]
        setattr(ranges, key, (lo_init + alpha * (lo_final - lo_init), hi_init + alpha * (hi_final - hi_init)))
    return alpha
