# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left/right mirror symmetry for R1, for rsl_rl's symmetry augmentation.

Week04 measured the trained policy's feet spending very different fractions of
the cycle in the air (left 0.416, right 0.646) despite stepping in clean
antiphase. Checking `R1.urdf` ruled out the model as the cause -- all 17
left/right link pairs match to 0 in mass, all six inertia components and centre
of mass -- so the asymmetry is a learned local optimum, and nothing in the
reward or the training loop ever asked for a symmetric gait.

This module supplies the mirror map so PPO can augment every minibatch with its
reflection (``RslRlSymmetryCfg(use_data_augmentation=True)``, Mittal et al.
2024). Under a reflection through the sagittal (x-z) plane:

- paired joints swap sides; ``waist_*`` and ``head_*`` map to themselves;
- a joint's sign flips iff its rotation axis is *not* the pitch axis. Verified
  against the URDF rather than inferred from names: every ``*_roll`` joint has
  axis (1,0,0), every ``*_yaw`` joint (0,0,1), and every ``*_pitch`` joint plus
  ``knee``/``elbow`` has (0,1,0). All 26 actuated joints are covered.
- vectors reflect componentwise: linear (x, -y, z), angular (-x, y, -z).

The joint permutation is built at runtime from ``robot.joint_names`` so it
cannot silently rot if Isaac Lab's articulation ordering changes.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# policy group, per frame: base_ang_vel(3) projected_gravity(3)
# velocity_commands(3) joint_pos(26) joint_vel(26) actions(24) = 85
# critic group adds base_lin_vel(3) base_incoming_wrench(6) body_friction(2) = 96
#
# Note joint_pos/joint_vel observe all 26 joints while the action vector covers
# only the 24 the policy drives (the head is excluded, see ActionsCfg) -- the two
# need different mirror maps, not just different widths.
_NUM_JOINTS = 26
_NUM_ACTIONS = 24

# Sign patterns for the 3-vectors, under reflection through the x-z plane.
_LINEAR = (1.0, -1.0, 1.0)  # a linear vector: y flips
_ANGULAR = (-1.0, 1.0, -1.0)  # a pseudovector: x and z flip

_MAP_CACHE: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _build_map(names: list[str], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(permutation, sign) for an ordered list of joint names.

    The permutation is *within this list*, so it must be built separately for
    every differently-ordered vector -- the 26-joint articulation order and the
    24-joint action order are not the same mapping.
    """
    index = {n: i for i, n in enumerate(names)}
    perm, sign = [], []
    for name in names:
        if name.startswith("left_"):
            partner = "right_" + name[len("left_") :]
        elif name.startswith("right_"):
            partner = "left_" + name[len("right_") :]
        else:
            partner = name  # waist_*, head_* are on the midline
        if partner not in index:
            raise ValueError(f"joint {name!r} has no mirror counterpart {partner!r}")
        perm.append(index[partner])
        # roll (axis x) and yaw (axis z) reverse under the reflection; pitch
        # (axis y), knee and elbow do not
        sign.append(-1.0 if ("_roll" in name or "_yaw" in name) else 1.0)

    return (
        torch.tensor(perm, dtype=torch.long, device=device),
        torch.tensor(sign, dtype=torch.float32, device=device),
    )


def _joint_map(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror map for the articulation's full joint vector (joint_pos/joint_vel)."""
    key = (id(env), "joints")
    if key not in _MAP_CACHE:
        _MAP_CACHE[key] = _build_map(
            list(env.unwrapped.scene["robot"].joint_names), env.unwrapped.device
        )
    return _MAP_CACHE[key]


def _action_map(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror map for the *action* vector.

    Separate from :func:`_joint_map` because the action term drives a subset of
    the joints (the head is excluded, see ActionsCfg), and its ordering is the
    articulation order restricted to that subset -- so index i means a different
    joint in the two vectors. Taken from the action term itself rather than
    reconstructed, so it stays correct if the actuated set changes again.
    """
    key = (id(env), "actions")
    if key not in _MAP_CACHE:
        term = env.unwrapped.action_manager.get_term("joint_pos")
        _MAP_CACHE[key] = _build_map(list(term._joint_names), env.unwrapped.device)
    return _MAP_CACHE[key]


def _mirror_joints(x: torch.Tensor, perm: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """Mirror a (..., 26) block of per-joint values."""
    return x[..., perm] * sign


def _mirror_vec3(x: torch.Tensor, pattern: tuple[float, float, float]) -> torch.Tensor:
    return x * torch.tensor(pattern, dtype=x.dtype, device=x.device)


# Observation terms in declaration order, as (width, kind). The order must match
# ObservationsCfg in flat_env_cfg.py -- these are the terms the manager
# concatenates, and the mirror is positional.
_POLICY_TERMS = [
    (3, "angular"),  # base_ang_vel
    (3, "linear"),  # projected_gravity
    (3, "command"),  # velocity_commands (lin_x, lin_y, ang_z)
    (_NUM_JOINTS, "joints"),  # joint_pos, relative to a mirror-symmetric default
    (_NUM_JOINTS, "joints"),  # joint_vel
    (_NUM_ACTIONS, "actions"),  # last_action -- driven joints only
]
_CRITIC_TERMS = _POLICY_TERMS + [
    (3, "linear"),  # base_lin_vel
    (6, "wrench"),  # base_incoming_wrench (force xyz, torque xyz)
    # body_friction is the mean over *both* feet, so swapping sides leaves it
    # unchanged -- identity is the correct mirror for an observation that
    # cannot distinguish left from right in the first place.
    (2, "identity"),
]

_COMMAND = (1.0, -1.0, -1.0)  # (lin_x, lin_y, ang_z): lateral and yaw reverse


def _mirror_block(block: torch.Tensor, kind: str, maps) -> torch.Tensor:
    """Mirror a (batch, history, width) block of one observation term."""
    if kind == "joints":
        return _mirror_joints(block, *maps["joints"])
    if kind == "actions":
        return _mirror_joints(block, *maps["actions"])
    if kind == "angular":
        return _mirror_vec3(block, _ANGULAR)
    if kind == "linear":
        return _mirror_vec3(block, _LINEAR)
    if kind == "command":
        return _mirror_vec3(block, _COMMAND)
    if kind == "wrench":
        return torch.cat(
            [_mirror_vec3(block[..., 0:3], _LINEAR), _mirror_vec3(block[..., 3:6], _ANGULAR)], dim=-1
        )
    if kind == "identity":
        return block
    raise ValueError(f"unknown mirror kind {kind!r}")


def _mirror_group(obs: torch.Tensor, terms, maps) -> torch.Tensor:
    """Mirror a whole observation group, history stacking included.

    The observation manager flattens history *per term* -- the group is
    ``[term0 x H frames][term1 x H frames]...``, not H copies of a full frame --
    so each term owns a contiguous ``width * H`` slice which reshapes to
    (batch, H, width). Every frame gets the same transform, so this is agnostic
    to whether the CircularBuffer runs oldest- or newest-first.
    """
    total = sum(w for w, _ in terms)
    if obs.shape[-1] % total != 0:
        raise ValueError(
            f"observation width {obs.shape[-1]} is not a multiple of the {total}-dim term"
            " layout this mirror assumes -- the observation terms changed, so the term"
            " table in tasks/r1_flat/symmetry.py needs updating too."
        )
    history = obs.shape[-1] // total

    pieces, offset = [], 0
    for width, kind in terms:
        span = width * history
        block = obs[:, offset : offset + span].view(obs.shape[0], history, width)
        pieces.append(_mirror_block(block, kind, maps).reshape(obs.shape[0], span))
        offset += span
    return torch.cat(pieces, dim=-1)


def mirror_obs_actions(
    env: "ManagerBasedRLEnv",
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str = "policy",
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """rsl_rl symmetry-augmentation hook: returns [original; mirrored] stacked.

    rsl_rl infers the augmentation factor from the returned batch size, so this
    must concatenate rather than replace.
    """
    maps = {"joints": _joint_map(env), "actions": _action_map(env)}

    aug_obs = None
    if obs is not None:
        terms = _CRITIC_TERMS if obs_type == "critic" else _POLICY_TERMS
        aug_obs = torch.cat([obs, _mirror_group(obs, terms, maps)], dim=0)

    aug_actions = None
    if actions is not None:
        aug_actions = torch.cat([actions, _mirror_joints(actions, *maps["actions"])], dim=0)

    return aug_obs, aug_actions
