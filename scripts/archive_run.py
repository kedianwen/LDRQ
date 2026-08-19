#!/usr/bin/env python
"""Archive a Week03+ rsl_rl training run into experiments/ for ablation comparisons.

Reads whatever's been logged so far under logs/rsl_rl/r1_flat/<run_id>/
(works on a still-running run, not just a finished one), and writes:
  - experiments/<run_id>/params/{env.yaml,agent.yaml}  (copied verbatim --
    the exact config, diff-able against another run)
  - experiments/<run_id>/pip_freeze.txt                (dependency versions,
    Week04's FR-T6 reproducibility requirement)
  - experiments/<run_id>/summary.md                    (final metrics pulled
    from that run's tensorboard event file)
  - a matching row appended (or updated) in experiments/runs.md

Doesn't need Isaac Sim -- just tensorboard's event-log reader -- so run it
with the conda env's plain python, not ~/IsaacLab/isaaclab.sh:

    conda activate env_isaaclab
    python scripts/archive_run.py [run_id]   # defaults to the most recent run

See experiments/README.md for the reasoning (why this is separate from the
git-ignored, much larger logs/ directory).
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TASK = "r1_flat"
_LOGS_DIR = _PROJECT_ROOT / "logs" / "rsl_rl" / _TASK
_EXPERIMENTS_DIR = _PROJECT_ROOT / "experiments"


def _pick_run_id() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    runs = sorted(p.name for p in _LOGS_DIR.iterdir() if p.is_dir())
    if not runs:
        raise SystemExit(f"No runs found under {_LOGS_DIR}")
    return runs[-1]


def _git_state() -> str:
    commit = subprocess.run(
        ["git", "-C", str(_PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip() or "(no commits yet)"
    dirty = subprocess.run(
        ["git", "-C", str(_PROJECT_ROOT), "status", "--porcelain"],
        capture_output=True, text=True,
    ).stdout.strip()
    return f"{commit}{' (dirty -- uncommitted changes present at archive time)' if dirty else ''}"


def _extract_block_weights(yaml_text: str, top_key: str) -> dict[str, float]:
    """Regex-extract {term_name: weight} from a `<top_key>:\\n  term:\\n    weight: X`
    block, without a full YAML parse (env.yaml contains !!python/tuple and
    !!python/object/apply:builtins.slice tags that yaml.safe_load rejects)."""
    lines = yaml_text.splitlines()
    start = next((i for i, l in enumerate(lines) if l == f"{top_key}:"), None)
    if start is None:
        return {}
    weights: dict[str, float] = {}
    current = None
    for line in lines[start + 1 :]:
        if line and not line.startswith(" "):
            break  # next top-level key
        m_term = re.match(r"^  (\w+):$", line)
        if m_term:
            current = m_term.group(1)
            continue
        m_weight = re.match(r"^    weight: (.+)$", line)
        if m_weight and current:
            weights[current] = float(m_weight.group(1))
    return weights


def _extract_command_ranges(yaml_text: str) -> dict[str, tuple[float, float]]:
    lines = yaml_text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip() == "ranges:"), None)
    if start is None:
        return {}
    ranges: dict[str, tuple[float, float]] = {}
    current = None
    nums: list[float] = []
    for line in lines[start + 1 :]:
        m_name = re.match(r"^\s{6}(\w+): !!python/tuple$", line)
        if m_name:
            if current and len(nums) == 2:
                ranges[current] = (nums[0], nums[1])
            current, nums = m_name.group(1), []
            continue
        m_val = re.match(r"^\s{6}- (-?[\d.]+)$", line)
        if m_val and current:
            nums.append(float(m_val.group(1)))
            continue
        if line.strip() and not line.startswith("      "):
            break
    if current and len(nums) == 2:
        ranges[current] = (nums[0], nums[1])
    return ranges


def _scalar_last(ea: EventAccumulator, tag: str):
    events = ea.Scalars(tag) if tag in ea.Tags().get("scalars", []) else []
    return events[-1] if events else None


def _scalar_first(ea: EventAccumulator, tag: str):
    events = ea.Scalars(tag) if tag in ea.Tags().get("scalars", []) else []
    return events[0] if events else None


def main() -> None:
    run_id = _pick_run_id()
    run_dir = _LOGS_DIR / run_id
    params_dir = run_dir / "params"
    if not (params_dir / "env.yaml").exists() or not (params_dir / "agent.yaml").exists():
        raise SystemExit(f"{params_dir} missing env.yaml/agent.yaml -- is {run_id} a real rsl_rl run dir?")

    env_yaml = (params_dir / "env.yaml").read_text()
    agent_yaml = (params_dir / "agent.yaml").read_text()

    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()

    reward_first = _scalar_first(ea, "Train/mean_reward")
    reward_last = _scalar_last(ea, "Train/mean_reward")
    eplen_last = _scalar_last(ea, "Train/mean_episode_length")
    fps_last = _scalar_last(ea, "Perf/total_fps")

    term_tags = sorted(t for t in ea.Tags().get("scalars", []) if t.startswith("Episode_Termination/"))
    reward_term_tags = sorted(t for t in ea.Tags().get("scalars", []) if t.startswith("Episode_Reward/"))

    reward_weights = _extract_block_weights(env_yaml, "rewards")
    command_ranges = _extract_command_ranges(env_yaml)
    m = re.search(r"^max_iterations: (\d+)", agent_yaml, re.M)
    max_iterations = m.group(1) if m else "?"
    m = re.search(r"^num_steps_per_env: (\d+)", agent_yaml, re.M)
    num_steps_per_env = m.group(1) if m else "?"
    m = re.search(r"^\s*num_envs: (\d+)", env_yaml, re.M)
    num_envs = m.group(1) if m else "?"
    m = re.search(r"^\s*dt: ([\d.]+)", env_yaml, re.M)
    sim_dt = m.group(1) if m else "?"

    out_dir = _EXPERIMENTS_DIR / run_id
    (out_dir / "params").mkdir(parents=True, exist_ok=True)
    shutil.copy(params_dir / "env.yaml", out_dir / "params" / "env.yaml")
    shutil.copy(params_dir / "agent.yaml", out_dir / "params" / "agent.yaml")

    # FR-T6: the configs pin what we set, this pins what we ran it against.
    # Uses whichever interpreter is running this script, so archive from inside
    # env_isaaclab or the snapshot describes the wrong environment.
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
    )
    (out_dir / "pip_freeze.txt").write_text(freeze.stdout or "(pip freeze failed)\n")

    last_it = reward_last.step if reward_last else "?"
    complete = f"{last_it}/{max_iterations}"

    lines = [
        f"# Run `{run_id}`",
        "",
        f"- Task: `Isaac-Velocity-Flat-R1-v0`, `num_envs={num_envs}`, `sim.dt={sim_dt}`, "
        f"`num_steps_per_env={num_steps_per_env}`",
        f"- Iterations completed: {complete}"
        # rsl_rl's iteration counter is 0-indexed, so the last logged `it` is
        # max_iterations - 1 on a run that finished normally.
        + (
            " (still running / did not finish -- re-archive later)"
            if max_iterations != "?" and isinstance(last_it, int) and last_it < int(max_iterations) - 1
            else ""
        ),
        f"- Git state at archive time: `{_git_state()}`",
        "",
        "## Final metrics",
        "",
        "| Metric | First logged | Last logged |",
        "|---|---|---|",
        f"| Train/mean_reward | {reward_first.value:.3f} (it {reward_first.step}) | "
        f"{reward_last.value:.3f} (it {reward_last.step}) |" if reward_first and reward_last else "| Train/mean_reward | - | - |",
        f"| Train/mean_episode_length | - | {eplen_last.value:.2f} (it {eplen_last.step}) |" if eplen_last else "| Train/mean_episode_length | - | - |",
        f"| Perf/total_fps | - | {fps_last.value:.0f} |" if fps_last else "| Perf/total_fps | - | - |",
        "",
        "## Reward weights used",
        "",
        "| Term | Weight |",
        "|---|---|",
    ]
    for name, w in reward_weights.items():
        lines.append(f"| {name} | {w:g} |")
    lines += ["", "## Command ranges used", "", "| Command | Range |", "|---|---|"]
    for name, (lo, hi) in command_ranges.items():
        lines.append(f"| {name} | [{lo:g}, {hi:g}] |")
    lines += ["", "## Final per-term breakdown (last logged value)", "", "| Term | Value |", "|---|---|"]
    for tag in reward_term_tags + term_tags:
        ev = _scalar_last(ea, tag)
        if ev:
            lines.append(f"| {tag} | {ev.value:.4f} |")
    lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines))

    reward_trend = (
        f"{reward_first.value:.2f} -> {reward_last.value:.2f}" if reward_first and reward_last else "?"
    )
    eplen_str = f"{eplen_last.value:.1f}" if eplen_last else "?"
    row = (
        f"| {run_id} | num_envs={num_envs} | it {complete} | "
        f"reward {reward_trend} | eplen {eplen_str} | [{run_id}]({run_id}/summary.md) |"
    )

    runs_md = _EXPERIMENTS_DIR / "runs.md"
    text = runs_md.read_text()
    marker = "<!-- archive_run.py inserts rows below this line -->"
    existing_row_re = re.compile(rf"^\|\s*{re.escape(run_id)}\s*\|.*\|$", re.M)
    if existing_row_re.search(text):
        text = existing_row_re.sub(row.replace("\\", "\\\\"), text)
    else:
        header = "| run_id | scale | progress | train reward trend | final ep length | details |\n|---|---|---|---|---|---|\n"
        if header.split("\n")[0] not in text:
            text = text.replace(marker, marker + "\n\n" + header + row)
        else:
            text = text.rstrip("\n") + "\n" + row + "\n"
    runs_md.write_text(text)

    print(f"[archive_run] wrote {out_dir / 'summary.md'}")
    print(f"[archive_run] updated {runs_md}")


if __name__ == "__main__":
    main()
