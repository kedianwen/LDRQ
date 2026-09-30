"""Assemble the demo video (BG-2): title card, captioned clips, the stage A figure, end card.

Every clip is normalised to 1280x720 @ 30 fps with its caption burnt in as a
lower third, and its audio track is DROPPED -- robot-side screen recordings pick
up the lab, and nothing in the demo needs sound. Cards and captions are drawn with
matplotlib (the ffmpeg shipped with Isaac Lab's imageio has no drawtext).

    /usr/bin/python3 scripts/make_demo_video.py --out outputs/demo/r1_demo.mp4 \\
        --clip "outputs/demo/sim_sequence.mp4::Simulation — the sweep's command sequence, kp_scale 1.0::1.5" \\
        --clip "docs/w07_walk_push_recovery.mp4::Robot — walking and recovering from pushes (onboard Orin NX, gantry slack)" \\
        --clip "<pg2 clip>::Robot — 60 s continuous walking, no gantry (PG-2)" \\
        --figure docs/stageA_kp_sweep.png

`--clip PATH::CAPTION[::SPEED]`, in order; SPEED > 1 plays faster (sim clips are
long and uneventful). The ffmpeg binary is found on PATH or inside the
env_isaaclab conda env's imageio_ffmpeg; `--ffmpeg` overrides.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

W, H, FPS = 1280, 720, 30
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def find_ffmpeg(explicit):
    if explicit:
        return explicit
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    hits = glob.glob(os.path.expanduser(
        "~/miniconda3/envs/env_isaaclab/lib/python3*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*"))
    if hits:
        return sorted(hits)[-1]
    raise SystemExit("[fail] no ffmpeg: pass --ffmpeg, or install one")


def card(path, title, lines=(), image=None):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    if image:
        img = plt.imread(image)
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.86])
        ax.imshow(img)
        ax.axis("off")
        fig.text(0.03, 0.935, title, fontsize=22, color=INK, va="center")
    else:
        fig.text(0.07, 0.58, title, fontsize=34, color=INK, va="bottom", weight="bold")
        for i, line in enumerate(lines):
            fig.text(0.07, 0.50 - i * 0.075, line, fontsize=19, color=INK2, va="top")
    fig.savefig(path, dpi=100, facecolor=SURFACE)
    plt.close(fig)


def lower_third(path, text):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_alpha(0.0)
    fig.patches.append(matplotlib.patches.Rectangle((0, 0), 1, 0.105, transform=fig.transFigure,
                                                    color="#000000", alpha=0.62, zorder=0))
    fig.text(0.03, 0.052, text, fontsize=20, color="#ffffff", va="center", zorder=1)
    fig.savefig(path, dpi=100, transparent=True)
    plt.close(fig)


def run(ff, args):
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y"] + args, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", action="append", default=[], help="PATH::CAPTION[::SPEED]")
    ap.add_argument("--figure", help="stage A figure, shown as a card")
    ap.add_argument("--figure_s", type=float, default=10.0)
    ap.add_argument("--ffmpeg")
    args = ap.parse_args()
    ff = find_ffmpeg(args.ffmpeg)
    tmp = Path(tempfile.mkdtemp(prefix="r1demo_"))
    parts = []
    norm = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,fps={FPS},format=yuv420p"

    def still(png, seconds, name):
        out = tmp / f"{len(parts):02d}_{name}.mp4"
        run(ff, ["-loop", "1", "-t", f"{seconds}", "-i", str(png), "-vf", norm, "-an",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "20", str(out)])
        parts.append(out)

    card(tmp / "title.png", "R1 · Sim2Real walking",
         ["Trained in Isaac Lab, run on the robot's own Orin NX (TensorRT, ROS 2).",
          "One question: how much actuator-gain error does the policy tolerate,",
          "in simulation and on the real robot?"])
    still(tmp / "title.png", 5, "title")

    for spec in args.clip:
        bits = spec.split("::")
        src, caption = bits[0], bits[1] if len(bits) > 1 else ""
        speed = float(bits[2]) if len(bits) > 2 else 1.0
        if not Path(src).is_file():
            raise SystemExit(f"[fail] clip not found: {src}")
        lt = tmp / f"lt{len(parts)}.png"
        lower_third(lt, caption)
        out = tmp / f"{len(parts):02d}_clip.mp4"
        vf = f"[0:v]setpts=PTS/{speed},{norm}[v];[v][1:v]overlay=0:0[o]"
        run(ff, ["-i", src, "-i", str(lt), "-filter_complex", vf, "-map", "[o]", "-an",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "20", str(out)])
        parts.append(out)

    if args.figure:
        card(tmp / "figure.png", "Stability domain in kp_scale: sim 0.80–2.00+, robot 1.10–1.50",
             image=args.figure)
        still(tmp / "figure.png", args.figure_s, "figure")

    card(tmp / "end.png", "What the robot taught us",
         ["The trained gain sits on the robot's lower stability edge.",
          "The sim2real gap is posture (lean), not joint tracking.",
          "Most real defects were silent: the chain reported success."])
    still(tmp / "end.png", 6, "end")

    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run(ff, ["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(out)])
    shutil.rmtree(tmp)
    print(f"[ok] wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(parts)} segments)")


if __name__ == "__main__":
    main()
