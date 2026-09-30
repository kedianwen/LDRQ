"""Stage A headline figure: the kp_scale stability domain, sim curve and real points on one axis.

Reads the simulated sweep (scripts/sweep_gain_robustness.py -> results.json) and
the real recordings (deploy/tools/probe_cpp/gain_sweep_real.py's own reduction,
imported, so the figure cannot disagree with ``--collect``). One metric per
panel, all sharing the kp_scale axis -- no second y-scale anywhere.

Needs only matplotlib (the system python has it; Isaac Sim is not involved):
    /usr/bin/python3 scripts/plot_gain_sweep.py \\
        --sim outputs/gain_sweep/<date>/results.json \\
        --real outputs/gain_sweep_real/2026-09-28/all \\
        --out docs/stageA_kp_sweep.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy" / "tools" / "probe_cpp"))
import gain_sweep_real as G  # noqa: E402

# Reference palette, categorical slots 1-2 (validated pair), neutral inks.
SIM = "#2a78d6"
REAL = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3de"
SURFACE = "#fcfcfb"


def real_rows(directory):
    """Per-recording verdicts. `directory` is either the recordings (~6 MB each, not
    in git) or the small json --dump-real wrote from them (docs/stageA_kp_sweep_real.json)."""
    if str(directory).endswith(".json"):
        d = json.loads(Path(directory).read_text())
        return d["recordings"], [tuple(x) for x in d["operator_stops"]]
    points, _ = G.load_recordings(directory, quiet=True)
    base = [m["leg_track_rms"] for kp, recs in points.items() if abs(kp - 1.0) < 1e-9 for _f, m in recs]
    baseline = min(base) if base else None
    rows = []
    for kp, recs in sorted(points.items()):
        for _f, m in recs:
            why = []
            if m["degraded"] or not m["have_status"]:
                why.append("status")
            if m["tilt_max_deg"] > G.TILT_LIMIT_DEG:
                why.append("tilt")
            if baseline and m["leg_track_rms"] > G.TRACKING_FACTOR * baseline:
                why.append("tracking")
            if m["tau_frac_max"] >= 1.0:
                why.append("torque")
            rows.append(dict(kp=kp, ok=not why, stand=m["stand_tilt_deg"], tilt=m["tilt_max_deg"],
                             ratio=m["leg_track_rms"] / baseline if baseline else float("nan"),
                             tau=100 * m["tau_frac_max"]))
    ops = G.read_operator_log(directory)
    return rows, ops


def domain(kps_ok, kps_bad):
    good = sorted(set(kps_ok) - set(kps_bad))
    return (min(good), max(good)) if good else None


def style(ax, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def threshold(ax, y, text):
    ax.axhline(y, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.995, y, text, transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=8, color=INK2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sim", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wide", action="store_true",
                    help="16:9 two-panel version (domains + max tilt) for slides and the demo video")
    ap.add_argument("--dump-real", metavar="JSON",
                    help="also write the reduced real-side verdicts, so the figure can be redrawn without the recordings")
    args = ap.parse_args()

    sim = json.loads(Path(args.sim).read_text())
    sp = sorted(sim["points"], key=lambda p: p["kp_scale"])
    sx = [p["kp_scale"] for p in sp]
    rows, ops = real_rows(args.real)
    if args.dump_real:
        Path(args.dump_real).write_text(json.dumps({
            "what": "real-robot kp_scale sweep, one entry per recording, reduced by gain_sweep_real.py "
                    "(commanded window; ratio = leg tracking rms / the kp=1.0 minimum)",
            "recordings": rows, "operator_stops": [list(o) for o in ops]}, indent=1))
        print(f"[ok] wrote {args.dump_real}")

    s_dom = domain([p["kp_scale"] for p in sp if p["stable"]], [p["kp_scale"] for p in sp if not p["stable"]])
    r_dom = domain([r["kp"] for r in rows if r["ok"]], [r["kp"] for r in rows if not r["ok"]] + [k for k, _ in ops])

    if args.wide:
        fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.2), sharex=True,
                                 gridspec_kw={"height_ratios": [0.8, 1], "hspace": 0.22})
        fig.subplots_adjust(top=0.9, bottom=0.12, left=0.07, right=0.87)
    else:
        fig, axes = plt.subplots(5, 1, figsize=(7.2, 10.2), sharex=True,
                                 gridspec_kw={"height_ratios": [0.9, 1, 1, 1, 1], "hspace": 0.28})
        fig.subplots_adjust(top=0.945, bottom=0.07)
    fig.patch.set_facecolor(SURFACE)

    # (a) the two domains -- the headline
    ax = axes[0]
    style(ax, "")
    ax.grid(False)
    s_open = s_dom and s_dom[1] >= max(sx)
    for y, dom, color, open_hi in ((1, s_dom, SIM, s_open), (0, r_dom, REAL, False)):
        if dom:
            ax.plot([dom[0], dom[1]], [y, y], color=color, linewidth=9, alpha=0.22, solid_capstyle="round")
            ax.text(2.13, y, f"{dom[0]:.2f}–{dom[1]:.2f}{'+' if open_hi else ''}\n"
                             f"width {'≥' if open_hi else ''}{dom[1] - dom[0]:.2f}",
                    va="center", fontsize=8.5, color=INK, linespacing=1.3)
    ax.scatter(sx, [1] * len(sx), s=46, zorder=3,
               facecolors=[SIM if p["stable"] else SURFACE for p in sp], edgecolors=SIM, linewidths=1.6)
    # repeats at one gain sit side by side vertically, so a pass and a fail both show
    seen, ry = {}, []
    for r in rows:
        n = sum(1 for q in rows if q["kp"] == r["kp"])
        i = seen.get(r["kp"], 0)
        seen[r["kp"]] = i + 1
        ry.append(0.0 if n == 1 else (i - (n - 1) / 2) * 0.28)
    ax.scatter([r["kp"] for r in rows], ry, s=46, zorder=3,
               facecolors=[REAL if r["ok"] else SURFACE for r in rows], edgecolors=REAL, linewidths=1.6)
    if ops:
        ax.scatter([k for k, _ in ops], [0] * len(ops), marker="x", s=60, color=REAL, linewidths=2, zorder=3)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["sim", "real"], fontsize=9, color=INK)
    ax.set_ylim(-0.6, 1.6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("kp_scale stability domain — filled = stable, hollow = not, × = operator e-stop (1.6, joint noise)",
                 fontsize=9, color=INK2, loc="left")

    # (b)-(e) one metric each; sim = median over envs, real = one dot per recording
    panels = [
        ("tilt_max_deg", "tilt", "max tilt (deg)", (G.TILT_LIMIT_DEG, "limit 20°")),
        ("stand_tilt_deg", "stand", "standing tilt (deg)", None),
        ("leg_track_ratio", "ratio", "leg tracking\n(× own kp=1.0)", (G.TRACKING_FACTOR, "limit 1.5×")),
        ("tau_frac_max", "tau", "peak torque\n(% of limit)", (100.0, "limit 100%")),
    ]
    # A gain at which every sim env fell while still standing has no commanded
    # window: its numbers would be read off a robot lying on the ground. Leave
    # them out and say why, rather than plot them as if measured.
    stood = [p for p in sp if not (p.get("fall_time_s") and p["fails"]["fell"] == p["n_envs"])]
    fell_x = [p["kp_scale"] for p in sp if p not in stood]
    for ax, (skey, rkey, ylabel, thr) in zip(axes[1:], panels):
        style(ax, ylabel)
        vx = [p["kp_scale"] for p in stood]
        sy = [100 * p[skey] if skey == "tau_frac_max" else p[skey] for p in stood]
        ax.plot(vx, sy, color=SIM, linewidth=2, zorder=2)
        ax.scatter(vx, sy, s=22, color=SIM, edgecolors=SURFACE, linewidths=1.2, zorder=3)
        if fell_x:
            ax.axvspan(min(fell_x) - 0.05, max(fell_x) + 0.05, color=SIM, alpha=0.06, linewidth=0, zorder=0)
        ax.scatter([r["kp"] for r in rows], [r[rkey] for r in rows], s=34, color=REAL,
                   edgecolors=SURFACE, linewidths=1.2, zorder=4)
        if thr:
            threshold(ax, *thr)
        ax.set_ylim(bottom=0)

    if fell_x:
        axes[1].text((min(fell_x) + max(fell_x)) / 2, 0.05, "sim: every env\nfalls while\nstanding",
                     transform=axes[1].get_xaxis_transform(), ha="center", va="bottom", fontsize=7.5, color=SIM)
    axes[1].plot([], [], color=SIM, linewidth=2, marker="o", markersize=4,
                 label="sim: median of 64 envs (no gantry, nominal)")
    axes[1].scatter([], [], s=34, color=REAL, label="real: one recording (slack gantry)")
    axes[1].legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK)

    axes[-1].set_xlabel("kp_scale (all joint stiffness × this; kd unchanged; 1.0 = as trained)",
                        color=INK2, fontsize=9)
    axes[-1].set_xlim(0.42, 2.08)
    axes[-1].set_xticks([round(0.5 + 0.1 * i, 1) for i in range(16)])
    for ax in axes:
        ax.axvline(1.0, color=GRID, linewidth=1.2, zorder=0)

    fig.suptitle("R1 · stage A · kp_scale sweep, same command sequence on both sides",
                 x=0.075, ha="left", fontsize=11, color=INK, y=0.985)
    if not args.wide:
        fig.text(0.075, 0.02,
                 "Scored over the commanded window of '" + sim["meta"]["sequence"] + "'. "
                 "Real torque is tau_est / rated; sim is pre-clip PD torque / effort limit.",
                 fontsize=7.5, color=INK2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"[ok] wrote {out}")
    print(f"     sim domain {s_dom}, real domain {r_dom}")


if __name__ == "__main__":
    main()
