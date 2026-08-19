# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Emit the deployment contract between the trained policy and the C++ runner.

Everything the robot-side code needs that is a property of the *trained policy*
rather than of the robot: joint ordering, observation term layout and history
depth, per-joint action scale, the default pose actions are offsets from, and
the control period.

None of this is recoverable from the ONNX file, which knows only "425 floats in,
24 floats out". Getting any of it wrong produces a policy that runs at full rate
and walks into the floor, so it is dumped from a live environment rather than
transcribed by hand:

- **joint order** is decided by PhysX when it parses the articulation, not by
  the order joints appear in the URDF or in the action term's regex list
  (``preserve_order: false``);
- **observation history is flattened per term**, not per frame -- see
  ``obs_assembler.hpp``;
- **action scale is per joint group**, so a single scalar is wrong.

Writes three files:
  deploy/ros2_ws/src/r1_policy_runner/config/policy_interface.yaml  (ROS params)
  deploy/interface/policy_interface.json                            (full contract)
  deploy/interface/policy_interface.md                              (for humans)

Run from the project root:
    ~/IsaacLab/isaaclab.sh -p deploy/tools/dump_interface.py --headless --num_envs 1
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

parser = argparse.ArgumentParser(description="Dump the R1 policy deployment interface.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-R1-Play-v0", help="Task to introspect.")
parser.add_argument("--num_envs", type=int, default=1, help="Only one env is needed.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import json  # noqa: E402

import tasks.r1_flat  # noqa: E402, F401  -- registers the task
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

_DEPLOY_ROOT = _PROJECT_ROOT / "deploy"


def _yaml_float_list(values) -> str:
    return "[" + ", ".join(f"{v:.6f}" for v in values) + "]"


def _yaml_str_list(values) -> str:
    return "[" + ", ".join(f'"{v}"' for v in values) + "]"


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]

    # -- observation layout -------------------------------------------------
    # group_obs_term_dim gives the *per-frame* shape of each term; the group's
    # history_length then multiplies it. Reading both from the manager (rather
    # than from the config) means this reflects what the policy actually saw.
    obs_mgr = env.observation_manager
    term_names = obs_mgr.active_terms["policy"]
    history = int(env_cfg.observations.policy.history_length or 1)

    # group_obs_term_dim already has the history folded in (a 3-wide term with
    # 5 frames reports 15), so divide it back out to get the per-frame width the
    # robot actually publishes each cycle.
    stacked_dims = [int(d[0]) for d in obs_mgr.group_obs_term_dim["policy"]]
    term_dims = []
    for name, stacked in zip(term_names, stacked_dims):
        if stacked % history:
            raise SystemExit(f"term '{name}' reports {stacked} dims, not divisible by history {history}")
        term_dims.append(stacked // history)

    frame_dim = sum(term_dims)
    obs_dim = int(obs_mgr.group_obs_dim["policy"][0])

    assert frame_dim * history == obs_dim, (
        f"per-term dims {term_dims} x history {history} = {frame_dim * history} "
        f"but the group reports {obs_dim}"
    )

    # -- action layout ------------------------------------------------------
    action_term = env.action_manager.get_term("joint_pos")
    action_joints = list(action_term._joint_names)
    action_ids = action_term._joint_ids
    if isinstance(action_ids, slice):
        action_ids = list(range(robot.num_joints))[action_ids]
    action_ids = [int(i) for i in action_ids]

    # _scale is a per-joint tensor once the grouped regex dict is resolved.
    scale = action_term._scale
    scale_list = [float(scale)] * len(action_joints) if scale.numel() == 1 else [
        float(v) for v in scale.reshape(-1)[: len(action_joints)]
    ]

    default_pos_all = robot.data.default_joint_pos[0].tolist()
    action_default = [float(default_pos_all[i]) for i in action_ids]

    all_joints = list(robot.joint_names)
    control_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)

    contract = {
        "run": "2026-08-19_11-03-32_week04_nohead",
        "task": args_cli.task,
        "control": {
            "sim_dt": float(env_cfg.sim.dt),
            "decimation": int(env_cfg.decimation),
            "control_dt": control_dt,
            "control_rate_hz": round(1.0 / control_dt, 6),
        },
        "observation": {
            "history_length": history,
            "frame_dim": frame_dim,
            "total_dim": obs_dim,
            "flatten": "per-term (term-major), oldest frame first within each term",
            "terms": [{"name": n, "width": d} for n, d in zip(term_names, term_dims)],
            "normalizer": "none (empirical_normalization=false) -- feed raw values",
        },
        "action": {
            "dim": len(action_joints),
            "joint_names": action_joints,
            "joint_ids_in_articulation": action_ids,
            "scale": scale_list,
            "default_pos": action_default,
            "target_formula": "q_target[i] = default_pos[i] + scale[i] * action[i]",
        },
        "articulation": {
            "num_joints": robot.num_joints,
            "joint_names": all_joints,
            "default_joint_pos": [float(v) for v in default_pos_all],
            "note": (
                "joint_pos / joint_vel observation terms cover ALL joints in this order; "
                "joint_pos is relative to default_joint_pos, joint_vel is absolute"
            ),
        },
    }

    # -- write -------------------------------------------------------------
    out_dir = _DEPLOY_ROOT / "interface"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "policy_interface.json").write_text(json.dumps(contract, indent=2) + "\n")

    cfg_dir = _DEPLOY_ROOT / "ros2_ws" / "src" / "r1_policy_runner" / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    yaml_lines = [
        "# GENERATED by deploy/tools/dump_interface.py -- do not edit by hand.",
        f"# source run: {contract['run']}",
        "#",
        "# Consumed by r1_policy_node. The node cross-checks obs_term_widths x",
        "# history_length against the engine's input dimension and refuses to start",
        "# if they disagree, so a stale copy of this file cannot silently mis-feed",
        "# the policy.",
        "/**:",
        "  ros__parameters:",
        f"    control_rate_hz: {contract['control']['control_rate_hz']}",
        f"    history_length: {history}",
        f"    obs_term_names: {_yaml_str_list(term_names)}",
        f"    obs_term_widths: [{', '.join(str(d) for d in term_dims)}]",
        "    # 24 actuated joints, articulation order (the head is not actuated by the policy)",
        f"    action_scale: {_yaml_float_list(scale_list)}",
        f"    default_joint_pos: {_yaml_float_list(action_default)}",
    ]
    (cfg_dir / "policy_interface.yaml").write_text("\n".join(yaml_lines) + "\n")

    md = [
        "# R1 policy deployment interface",
        "",
        f"Generated from `{contract['run']}` by `deploy/tools/dump_interface.py`.",
        "",
        "## Control loop",
        "",
        f"- sim dt `{contract['control']['sim_dt']}` x decimation `{contract['control']['decimation']}`"
        f" = **{control_dt * 1000:.1f} ms** per control step (**{contract['control']['control_rate_hz']:.1f} Hz**)",
        "",
        "## Observation (policy input)",
        "",
        f"**{obs_dim} floats** = {history} frames x {frame_dim} per frame, "
        "flattened **per term** (term-major), oldest frame first inside each term.",
        "",
        "| term | width/frame | x history | offset |",
        "|---|---|---|---|",
    ]
    off = 0
    for n, d in zip(term_names, term_dims):
        md.append(f"| `{n}` | {d} | {d * history} | {off} |")
        off += d * history
    md += [
        "",
        "No input normalisation: the policy was trained with "
        "`empirical_normalization=false`, so the exported graph is the whole "
        "computation. Feed raw values.",
        "",
        "## Action (policy output)",
        "",
        f"**{len(action_joints)} floats**, articulation order, mapped as "
        "`q_target[i] = default_pos[i] + scale[i] * action[i]`.",
        "",
        "| # | joint | scale | default (rad) |",
        "|---|---|---|---|",
    ]
    for i, (n, s, d) in enumerate(zip(action_joints, scale_list, action_default)):
        md.append(f"| {i} | `{n}` | {s:.4f} | {d:.4f} |")
    md += [
        "",
        f"## Articulation joint order ({robot.num_joints} joints)",
        "",
        "The `joint_pos` / `joint_vel` observation terms cover **all** joints in "
        "this order -- including the two head joints the policy does not drive. "
        "`joint_pos` is relative to the default pose; `joint_vel` is absolute.",
        "",
        "| # | joint | default (rad) | policy-actuated |",
        "|---|---|---|---|",
    ]
    for i, n in enumerate(all_joints):
        md.append(f"| {i} | `{n}` | {default_pos_all[i]:.4f} | {'yes' if i in action_ids else 'NO'} |")
    (out_dir / "policy_interface.md").write_text("\n".join(md) + "\n")

    print(f"[ok] observation {obs_dim} = {history} x {frame_dim}  terms={list(zip(term_names, term_dims))}")
    print(f"[ok] action {len(action_joints)} joints, articulation has {robot.num_joints}")
    print(f"[ok] control {control_dt * 1000:.1f} ms ({contract['control']['control_rate_hz']:.1f} Hz)")
    print(f"[ok] wrote {out_dir/'policy_interface.json'}")
    print(f"[ok] wrote {out_dir/'policy_interface.md'}")
    print(f"[ok] wrote {cfg_dir/'policy_interface.yaml'}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
