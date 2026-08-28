# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the M1 gate-review figures from data already on disk.

The Week04 plan asks for "DR 对曲线/表现的影响（有/无 DR 对比一张图）" -- a
figure, not a table. This script builds it from the baseline reports that
``eval_baseline.py`` already wrote and the tensorboard scalars the training runs
already logged, so the figure is regenerated rather than redrawn by hand and
cannot drift away from the numbers it claims to show.

Two figures:

``m1_dr_robustness.png``
    The headline. Tracking error against commanded speed for the Week03 policy
    (trained without domain randomization) and the final Week04 policy, each
    under nominal and stress conditions, plus fall counts and energy.

``m1_training_curves.png``
    Mean reward against iteration for the three Week04 runs. These three share a
    reward function, so the curves are directly comparable; the Week03 runs are
    deliberately absent because the reward terms changed between weeks and
    overlaying them would invite a comparison the numbers do not support.

Run from the project root (needs matplotlib + tensorboard, both in env_isaaclab):
    conda run -n env_isaaclab python scripts/plot_m1_figures.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOG_ROOT = _PROJECT_ROOT / "logs" / "rsl_rl" / "r1_flat"
_OUT_DIR = _PROJECT_ROOT / "docs"

# Runs referenced below. Kept as one table so a reader can see at a glance which
# checkpoint every number in the figures comes from.
WEEK03 = "2026-08-15_17-34-57_touchdown_gate"
WEEK04_DR = "2026-08-17_17-38-37_week04_dr"
WEEK04_HWSPEC = "2026-08-18_15-33-21_week04_hwspec"
WEEK04_NOHEAD = "2026-08-19_11-03-32_week04_nohead"

# week04_dr was trained as train-to-300 then resume-to-3000, so its curve lives
# in two event files. Both are needed or the plot silently starts at iteration
# 300 with a spike -- the resumed run's reward buffer starts empty and refills
# over the first few iterations, which looks like a training collapse and is
# not one.
WEEK04_DR_PHASE1 = "2026-08-17_17-18-51_week04_dr"

PG1_THRESHOLD = 0.15
PG1_RANGE = (0.5, 1.0)

# Iterations discarded after a resume while rsl_rl's reward buffer refills.
_RESUME_WARMUP = 25

# Instrument palette: teal for the policy that carries DR, warm grey-red for the
# one that does not. Nominal is solid, stress dashed -- so the reader compares
# line colour for "which policy" and line style for "which condition".
C_NODR = "#B4471F"
C_DR = "#0B6255"
C_GRID = "#C9D2D0"
C_TEXT = "#12181B"
C_RULE = "#8A9694"


def _use_cjk_font() -> None:
    """Pick a font that can actually render the Chinese labels.

    Without this matplotlib silently substitutes DejaVu Sans and every CJK
    glyph becomes a tofu box -- a failure that only shows up in the rendered
    PNG, which is exactly the kind of thing that reaches a slide unnoticed.
    """
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ("Noto Sans CJK JP", "Noto Sans CJK SC", "Droid Sans Fallback", "WenQuanYi Zen Hei"):
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[font] using {candidate}")
            return
    raise SystemExit("no CJK-capable font found; install fonts-noto-cjk or relabel the figure in English")


# ---------------------------------------------------------------------------
# Parsing what is already on disk
# ---------------------------------------------------------------------------

_ROW = re.compile(r"^\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(.+?)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|$")


def read_baseline(run: str, filename: str) -> dict:
    """Pull the per-speed table out of an eval_baseline.py report.

    Handles both column layouts the script has emitted: Week03's reports carry a
    ``survival`` column, later ones carry ``falls / episodes``.
    """
    path = _LOG_ROOT / run / filename
    text = path.read_text()

    speeds, errors, energies = [], [], []
    for line in text.splitlines():
        m = _ROW.match(line.strip())
        if not m:
            continue
        speeds.append(float(m.group(1)))
        errors.append(float(m.group(2)))
        energies.append(float(m.group(4)))

    if not speeds:
        raise SystemExit(f"no data rows parsed from {path}")

    falls = None
    m = re.search(r"falls across all speeds:\s*(\d+)\s*/\s*(\d+)", text)
    if m:
        falls = (int(m.group(1)), int(m.group(2)))

    return {"speeds": speeds, "errors": errors, "energies": energies, "falls": falls, "path": path}


def read_reward_curve(runs: str | list[str], tag: str = "Train/mean_reward") -> tuple[list[int], list[float]]:
    """Read one scalar series, concatenating across a train-then-resume pair.

    Later phases win on overlapping steps, and the first few iterations after a
    resume are dropped: rsl_rl restarts its reward buffer empty, so those points
    report a partially-filled average that dips hard and recovers. Plotting them
    would show a collapse the training never had.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    if isinstance(runs, str):
        runs = [runs]

    merged: dict[int, float] = {}
    for phase, run in enumerate(runs):
        events = sorted((_LOG_ROOT / run).glob("events.out.tfevents.*"))
        if not events:
            raise SystemExit(f"no event file under {_LOG_ROOT / run}")
        acc = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
        acc.Reload()
        if tag not in acc.Tags()["scalars"]:
            raise SystemExit(f"tag '{tag}' not in {run}; available: {acc.Tags()['scalars'][:12]}")
        series = acc.Scalars(tag)
        if phase > 0:
            series = series[_RESUME_WARMUP:]
        for s in series:
            merged[s.step] = s.value

    steps = sorted(merged)
    return steps, [merged[s] for s in steps]


def _style(ax) -> None:
    ax.grid(True, color=C_GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_RULE)
    ax.tick_params(colors=C_TEXT, labelsize=9)


# ---------------------------------------------------------------------------
# Figure 1 -- robustness
# ---------------------------------------------------------------------------

def figure_robustness() -> Path:
    w03_nom = read_baseline(WEEK03, "baseline_week03_nodr.md")
    w03_str = read_baseline(WEEK03, "baseline_week03_stress.md")
    w04_nom = read_baseline(WEEK04_NOHEAD, "baseline_nohead_nominal.md")
    w04_str = read_baseline(WEEK04_NOHEAD, "baseline_nohead_stress.md")
    dr_str = read_baseline(WEEK04_DR, "baseline_week04_stress.md")

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    fig.patch.set_facecolor("white")

    # -- (a) tracking error -------------------------------------------------
    ax = axes[0]
    ax.axhspan(0, PG1_THRESHOLD, color=C_DR, alpha=0.05)
    ax.axhline(PG1_THRESHOLD, color=C_RULE, linestyle=":", linewidth=1.4)
    ax.text(0.31, PG1_THRESHOLD * 1.06, f"PG-1 阈值 {PG1_THRESHOLD}", fontsize=8.5, color=C_TEXT)

    ax.plot(w03_nom["speeds"], w03_nom["errors"], "-o", color=C_NODR, ms=5, lw=1.9, label="Week03 无DR · 标称")
    ax.plot(w03_str["speeds"], w03_str["errors"], "--s", color=C_NODR, ms=5, lw=1.9, alpha=0.75,
            label="Week03 无DR · 扰动")
    ax.plot(w04_nom["speeds"], w04_nom["errors"], "-o", color=C_DR, ms=5, lw=2.2, label="Week04 有DR · 标称")
    ax.plot(w04_str["speeds"], w04_str["errors"], "--s", color=C_DR, ms=5, lw=2.2, alpha=0.85,
            label="Week04 有DR · 扰动")

    ax.set_xlabel("指令前进速度 (m/s)", fontsize=10)
    ax.set_ylabel("线速度跟踪误差 (m/s)", fontsize=10)
    ax.set_title("(a) 跟踪精度：扰动下才见分晓", fontsize=11.5, fontweight="bold", loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.set_ylim(0, 0.42)
    _style(ax)

    # -- (b) falls ----------------------------------------------------------
    ax = axes[1]
    labels = ["Week03\n无DR", "Week04\n初版", "Week04\n最终版"]
    values = [w03_str["falls"][0], dr_str["falls"][0], w04_str["falls"][0]]
    totals = [w03_str["falls"][1], dr_str["falls"][1], w04_str["falls"][1]]
    bars = ax.bar(labels, values, color=[C_NODR, "#C9895F", C_DR], width=0.6)
    for bar, v, t in zip(bars, values, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 14, f"{v} / {t}",
                ha="center", fontsize=10, fontweight="bold", color=C_TEXT)
    ax.set_ylabel("扰动条件下的摔倒回合数", fontsize=10)
    ax.set_title("(b) 存活：无DR 策略每一个回合都摔", fontsize=11.5, fontweight="bold", loc="left")
    ax.set_ylim(0, max(values) * 1.18)
    _style(ax)

    # -- (c) energy ---------------------------------------------------------
    ax = axes[2]
    ax.plot(w03_nom["speeds"], w03_nom["energies"], "-o", color=C_NODR, ms=5, lw=1.9, label="Week03 无DR")
    ax.plot(w04_nom["speeds"], w04_nom["energies"], "-o", color=C_DR, ms=5, lw=2.2, label="Week04 最终版")
    drop = 1 - w04_nom["energies"][-1] / w03_nom["energies"][-1]
    ax.annotate(f"1.0 m/s 处 −{drop * 100:.0f}%",
                xy=(1.0, w04_nom["energies"][-1]), xytext=(0.62, w04_nom["energies"][-1] + 55),
                fontsize=9, color=C_TEXT,
                arrowprops=dict(arrowstyle="->", color=C_RULE, lw=1.2))
    ax.set_xlabel("指令前进速度 (m/s)", fontsize=10)
    ax.set_ylabel("能耗 (W)", fontsize=10)
    ax.set_title("(c) 能耗：标称条件下同样改善", fontsize=11.5, fontweight="bold", loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    _style(ax)

    fig.suptitle("M1 · 域随机化的作用：Week03（无DR）对比 Week04 最终策略",
                 fontsize=13.5, fontweight="bold", x=0.007, ha="left", y=0.995)
    fig.text(0.007, 0.005,
             "扰动条件 = 地面摩擦 0.6 + 每 5s ±0.6 m/s 侧推。每档 64 环境 × 500 控制步。"
             "注意：Week03→Week04 除 DR 外还改动了观测历史、执行器规格、动作空间与指令课程，"
             "因此这不是单变量消融；严格的 DR 消融是 W11 的排期内容。",
             fontsize=8, color="#55625F", ha="left")

    fig.tight_layout(rect=(0, 0.045, 1, 0.955))
    out = _OUT_DIR / "m1_dr_robustness.png"
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2 -- training curves
# ---------------------------------------------------------------------------

def figure_training() -> Path:
    runs = [
        ([WEEK04_DR_PHASE1, WEEK04_DR], "week04_dr　执行器 150N·m（超硬件规格）", "#C9895F"),
        ([WEEK04_HWSPEC], "week04_hwspec　执行器对齐硬件 + 对称性增强", "#5B8A9A"),
        ([WEEK04_NOHEAD], "week04_nohead　头部移出动作空间（最终版）", C_DR),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    fig.patch.set_facecolor("white")

    finals = []
    for run, label, colour in runs:
        steps, values = read_reward_curve(run)
        ax.plot(steps, values, lw=2.0, color=colour, label=label)
        finals.append((steps[-1], values[-1], colour))

    for x, y, colour in finals:
        ax.plot([x], [y], "o", color=colour, ms=6)
        ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(6, -3), textcoords="offset points",
                    fontsize=9.5, fontweight="bold", color=colour)

    ax.axvline(1500, color=C_RULE, linestyle=":", lw=1.2)
    ax.text(1520, ax.get_ylim()[0] + 1.2, "it1500", fontsize=8.5, color="#55625F")

    ax.set_xlabel("迭代", fontsize=10)
    ax.set_ylabel("平均回合回报", fontsize=10)
    ax.set_title("Week04 三次训练的学习曲线（奖励函数相同，可直接比较）",
                 fontsize=12.5, fontweight="bold", loc="left")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    _style(ax)

    fig.text(0.01, 0.038,
             "两次修正都让训练更快而非更慢：执行器降到硬件规格后控制器不再饱和，"
             "去掉两个对任务无贡献的自由度后探索空间缩小。",
             fontsize=8.5, color="#55625F", ha="left")
    fig.text(0.01, 0.008,
             "week04_dr 为「训练 300 轮 + 续训」两阶段，曲线由两个日志拼接，"
             "续训后前 25 轮已剔除（rsl_rl 的回报缓冲重启后为空）。"
             "Week03 曲线未叠加：奖励项在两周之间变过，不可比。",
             fontsize=8.5, color="#55625F", ha="left")

    fig.tight_layout(rect=(0, 0.075, 1, 1))
    out = _OUT_DIR / "m1_training_curves.png"
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    _use_cjk_font()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (figure_robustness(), figure_training()):
        print(f"[ok] wrote {path.relative_to(_PROJECT_ROOT)} ({path.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
