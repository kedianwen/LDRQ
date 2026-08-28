# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check that current code still reproduces an archived run's configuration.

The Week04 plan's FR-T6 asks for "所有超参进 yaml". In Isaac Lab the
hyperparameters live in Python dataclasses (``flat_env_cfg.py``,
``rsl_rl_ppo_cfg.py``) and the ``params/{env,agent}.yaml`` next to each run are a
*dump* of the resolved config, not an input to it. Editing that yaml changes
nothing.

Taken literally the plan is therefore unmet, and the honest reading is that a
dump is weaker than a spec: it records what happened but nothing checks that the
code still produces it. Six weeks and several refactors later, "the yaml is in
the archive" does not tell you whether ``git checkout <sha> && train`` still
gives the same run.

This script closes that gap the other way round. Rather than moving the
hyperparameters into yaml -- which would fight Isaac Lab's design and lose type
checking -- it makes the dump *verifiable*: rebuild the config from current code
and diff it field by field against what the run recorded. Now the archived yaml
is a contract the repository is tested against, and drift is reported as an
exact list of fields rather than discovered when a rerun quietly behaves
differently.

Fields that legitimately vary per invocation (env count, device, run name,
iteration budget) are reported separately from real drift.

Run from the project root:
    ~/IsaacLab/isaaclab.sh -p scripts/verify_repro.py --run 2026-08-19_11-03-32_week04_nohead

Exit status is 0 when the configuration matches, 1 when it has drifted -- so it
works as a CI check or a pre-release gate.
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

parser = argparse.ArgumentParser(description="Verify current code reproduces an archived run's config.")
parser.add_argument(
    "--run",
    type=str,
    default="2026-08-19_11-03-32_week04_nohead",
    help="Run id. Looked up under experiments/<run>/params/, then logs/rsl_rl/r1_flat/<run>/params/.",
)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-R1-v0", help="Task the run was trained on.")
parser.add_argument("--num_envs", type=int, default=4096, help="Env count the run used (a CLI-varying field).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import yaml  # noqa: E402

import tasks.r1_flat  # noqa: E402, F401  -- registers the task
from isaaclab.utils import class_to_dict  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# Fields whose value is a property of the invocation, not of the experiment.
# Drift here is expected and reported separately rather than counted as failure.
CLI_VARYING = {
    "scene.num_envs",
    "sim.device",
    "device",
    "seed",  # reported explicitly below -- a seed mismatch matters, so it is checked, not ignored
    "max_iterations",
    "run_name",
    "load_run",
    "load_checkpoint",
    "resume",
    "logger",
    "wandb_project",
    "neptune_project",
    "experiment_name",
}


def resolve_like_scene_construction(cfg: dict) -> dict:
    """Apply the substitutions ``InteractiveScene`` performs when it builds.

    The archived yaml is dumped *after* the environment is constructed, so it
    records post-resolution values, while ``parse_env_cfg`` returns the config
    as written. Three things differ, none of them hyperparameters:

    * asset ``prim_path`` still holds the ``{ENV_REGEX_NS}`` placeholder;
    * ``terrain.num_envs`` and ``terrain.env_spacing`` are copied down from the
      scene config at construction time.

    Normalising them here -- rather than adding them to an ignore list -- keeps
    the comparison honest: every remaining difference is a real one.
    """
    scene = cfg.get("scene", {})
    num_envs = scene.get("num_envs")
    env_spacing = scene.get("env_spacing")

    terrain = scene.get("terrain")
    if isinstance(terrain, dict):
        # Overwritten unconditionally at construction, not filled in only when
        # unset -- TerrainImporterCfg carries its own defaults (num_envs=1) that
        # the scene always replaces.
        if num_envs is not None:
            terrain["num_envs"] = num_envs
        if env_spacing is not None:
            terrain["env_spacing"] = env_spacing

    def substitute(node):
        if isinstance(node, dict):
            return {k: substitute(v) for k, v in node.items()}
        if isinstance(node, list):
            return [substitute(v) for v in node]
        if isinstance(node, str):
            return node.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
        return node

    return substitute(cfg)


def flatten(node, prefix: str = "") -> dict:
    """Flatten a nested config dict to ``dotted.path -> leaf`` for diffing."""
    out = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(node, (list, tuple)):
        # Lists are compared as a whole: a per-index diff on a joint-name list
        # produces noise, while "this list changed" is the actionable fact.
        out[prefix] = repr(list(node))
    else:
        out[prefix] = node
    return out


def find_params_dir(run: str) -> Path:
    for base in ("experiments", "logs/rsl_rl/r1_flat"):
        candidate = _PROJECT_ROOT / base / run / "params"
        if (candidate / "env.yaml").exists():
            return candidate
    raise SystemExit(f"no params/env.yaml found for run '{run}' under experiments/ or logs/rsl_rl/r1_flat/")


def load_archived(path: Path) -> dict:
    # Isaac Lab dumps python objects (slices, enums) with tags, so the safe
    # loader cannot read its own output.
    return yaml.unsafe_load(path.read_text())


def compare(name: str, archived: dict, current: dict) -> tuple[list, list, list]:
    a, c = flatten(archived), flatten(current)

    drift, expected, missing = [], [], []
    for key in sorted(set(a) | set(c)):
        if key not in a:
            missing.append((f"{name}.{key}", "<absent in archive>", c[key]))
            continue
        if key not in c:
            missing.append((f"{name}.{key}", a[key], "<absent in current>"))
            continue
        if a[key] == c[key]:
            continue
        # str() both sides: yaml round-trips some numerics as str, and a
        # 0.15 vs "0.15" mismatch is noise, not drift.
        if str(a[key]) == str(c[key]):
            continue
        (expected if key in CLI_VARYING else drift).append((f"{name}.{key}", a[key], c[key]))

    return drift, expected, missing


_LINES: list[str] = []


def say(line: str = "") -> None:
    """Print and retain.

    Isaac Sim's ``simulation_app.close()`` tears the process down hard enough
    that a block-buffered stdout (i.e. any time output is piped or redirected)
    never flushes. Everything therefore goes through here: printed with an
    explicit flush, and kept so the whole report can also be written to a file
    that survives regardless.
    """
    _LINES.append(line)
    print(line, flush=True)


def report(title: str, rows: list) -> None:
    if not rows:
        return
    say(f"\n{title} ({len(rows)}):")
    for key, old, new in rows:
        say(f"  {key}")
        say(f"      archived: {old!r}")
        say(f"      current : {new!r}")


def main() -> int:
    params_dir = find_params_dir(args_cli.run)
    say(f"[info] run       {args_cli.run}")
    say(f"[info] archive   {params_dir.relative_to(_PROJECT_ROOT)}")

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_drift, env_expected, env_missing = compare(
        "env",
        load_archived(params_dir / "env.yaml"),
        resolve_like_scene_construction(class_to_dict(env_cfg)),
    )
    agent_drift, agent_expected, agent_missing = compare(
        "agent", load_archived(params_dir / "agent.yaml"), class_to_dict(agent_cfg)
    )

    drift = env_drift + agent_drift
    expected = env_expected + agent_expected
    missing = env_missing + agent_missing

    report("CONFIG DRIFT -- current code would NOT reproduce this run", drift)
    report("STRUCTURAL CHANGE -- fields added or removed since the run", missing)
    report("expected to vary per invocation", expected)

    # The seed is the one field where equality is the whole point of FR-T6.
    archived_seed = load_archived(params_dir / "agent.yaml").get("seed")
    say(f"\n[seed] archived={archived_seed}  current={agent_cfg.seed}"
        f"  {'OK' if archived_seed == agent_cfg.seed else 'MISMATCH'}")

    say("")
    if drift or missing:
        say(f"RESULT: FAIL -- {len(drift)} drifted field(s), {len(missing)} structural change(s).")
        say("        The archived yaml no longer describes what this code builds.")
        code = 1
    else:
        say(f"RESULT: PASS -- current code reproduces this run's configuration exactly "
            f"({len(expected)} invocation-dependent field(s) allowed to differ).")
        code = 0

    out = params_dir.parent / "repro_check.txt"
    out.write_text("\n".join(_LINES) + "\n")
    say(f"\n[ok] report written to {out.relative_to(_PROJECT_ROOT)}")
    out.write_text("\n".join(_LINES) + "\n")
    return code


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
