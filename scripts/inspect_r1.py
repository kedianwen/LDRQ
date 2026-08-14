#!/usr/bin/env python3
"""Load R1 in an empty scene, let it settle under PD hold, and produce Week01's
joint/limit/mass verification table + a standing screenshot.

Run from the project root::

    ~/IsaacLab/isaaclab.sh -p scripts/inspect_r1.py

Output:
    docs/joint_check.md   -- per-joint limits / drive gains / mass sanity table
    docs/r1_standing.png  -- screenshot of R1 after settling
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect the R1 articulation: limits, masses, standing check.")
parser.add_argument(
    "--settle-steps",
    type=int,
    default=5500,
    help="Physics steps to let R1 settle under PD hold. Default is ~11s at dt=0.002 (Week02's >=10s check).",
)
parser.add_argument(
    "--trace-every", type=int, default=250, help="Print height/roll/pitch every N steps (0 to disable)."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Need offscreen render to grab a screenshot even with no display attached.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----- after the app is up -----
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import convert_dict_to_backend  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "assets" / "r1"))
from r1 import R1_CFG, require_usd  # noqa: E402


def main() -> None:
    require_usd()  # fail fast with a clear message if usd/R1.usd hasn't been built yet

    out_dir = _PROJECT_ROOT / "docs"
    out_dir.mkdir(exist_ok=True)

    # dt=0.002 (500Hz): required at R1's standing-gains (assets/r1/r1.py,
    # hip/knee/ankle stiffness ~1200 N*m/rad). The default 1/60s is too coarse
    # for PhysX's implicit joint-drive solver at this stiffness -- R1 fell
    # identically regardless of gains until this was fixed (see r1.py and
    # scripts/README.md for the full diagnosis).
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.002)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 2.0, 1.3], [0.0, 0.0, 0.7])

    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot_cfg = R1_CFG.copy()
    robot_cfg.prim_path = "/World/R1"
    robot = Articulation(cfg=robot_cfg)

    camera_cfg = CameraCfg(
        prim_path="/World/CameraSensor",
        update_period=0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    camera = Camera(cfg=camera_cfg)

    sim.reset()
    camera.set_world_poses_from_view(
        torch.tensor([[2.0, 2.0, 1.3]], device=sim.device),
        torch.tensor([[0.0, 0.0, 0.7]], device=sim.device),
    )
    print("[INFO] R1 spawned. Settling under PD hold...")

    dt = sim.get_physics_dt()
    default_joint_pos = robot.data.default_joint_pos.clone()
    height_before = float(robot.data.root_pos_w[0, 2].item())

    for step in range(args_cli.settle_steps):
        robot.set_joint_position_target(default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        camera.update(dt)
        if args_cli.trace_every and step % args_cli.trace_every == 0:
            h = float(robot.data.root_pos_w[0, 2].item())
            q = robot.data.root_quat_w[0].detach().cpu().numpy()
            w, x, y, z = q
            roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
            pitch = np.degrees(np.arcsin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
            print(f"[TRACE] step={step} t={step*dt:.3f}s height={h:.3f} roll={roll:.1f} pitch={pitch:.1f}", flush=True)

    # A few extra steps purely to let the camera's render buffer populate.
    for _ in range(5):
        sim.step()
        camera.update(dt)

    height_after = float(robot.data.root_pos_w[0, 2].item())
    quat = robot.data.root_quat_w[0].detach().cpu().numpy()  # (w, x, y, z)
    # roll/pitch from quaternion (small-angle-safe formula)
    w, x, y, z = quat
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    tilt_deg = np.degrees(max(abs(roll), abs(pitch)))
    height_drop = height_before - height_after
    stable = height_drop < 0.05 and tilt_deg < 15.0

    print(f"[INFO] Base height before/after settle: {height_before:.3f} -> {height_after:.3f} m")
    print(f"[INFO] Max roll/pitch tilt after settle: {tilt_deg:.2f} deg")
    print(f"[INFO] Standing stable (height drop < 5cm, tilt < 15deg): {stable}")

    # ---- joint table ----
    names = robot.data.joint_names
    n = len(names)
    pos_limits = robot.data.joint_pos_limits[0].detach().cpu().numpy()  # (n, 2) [lower, upper]
    effort_limits = robot.data.joint_effort_limits[0].detach().cpu().numpy()
    stiffness = robot.data.joint_stiffness[0].detach().cpu().numpy()
    damping = robot.data.joint_damping[0].detach().cpu().numpy()
    default_pos = default_joint_pos[0].detach().cpu().numpy()

    lines = [
        "# R1 Joint / Limit / Drive Verification (Week01 exit criterion)",
        "",
        f"- Total DOF: **{n}** (expected 26 actuated)",
        f"- Base standing height before/after {args_cli.settle_steps} settle steps: "
        f"{height_before:.3f} m -> {height_after:.3f} m (drop {height_drop*1000:.1f} mm)",
        f"- Max roll/pitch after settle: {tilt_deg:.2f} deg",
        f"- **Standing stable: {'YES' if stable else 'NO'}**",
        "",
        "| # | joint | lower (rad) | upper (rad) | default pos (rad) | effort limit (N·m) | stiffness | damping |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, name in enumerate(names):
        lines.append(
            f"| {i} | {name} | {pos_limits[i, 0]:.3f} | {pos_limits[i, 1]:.3f} | "
            f"{default_pos[i]:.3f} | {effort_limits[i]:.1f} | {stiffness[i]:.1f} | {damping[i]:.1f} |"
        )

    # ---- mass sanity check ----
    body_names = robot.data.body_names
    masses = robot.data.default_mass[0].detach().cpu().numpy()
    zero_mass_bodies = [body_names[i] for i in range(len(body_names)) if masses[i] <= 0.0]
    total_mass = float(masses.sum())

    lines += [
        "",
        f"## Mass sanity ({len(body_names)} bodies, total {total_mass:.2f} kg)",
        "",
        f"- Bodies with zero/negative mass: **{zero_mass_bodies if zero_mass_bodies else 'none'}**",
    ]

    # ---- limit sanity (flag anything degenerate) ----
    degenerate = [names[i] for i in range(n) if pos_limits[i, 1] - pos_limits[i, 0] < 1e-6]
    lines += [
        "",
        f"## Joint limit sanity",
        "",
        f"- Joints with degenerate (zero-width) position limits: **{degenerate if degenerate else 'none'}**",
    ]

    report_path = out_dir / "joint_check.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"[INFO] Wrote joint/limit/mass table to {report_path}")

    # ---- screenshot ----
    camera.update(dt=dt)
    if "rgb" in camera.data.output:
        rgb = convert_dict_to_backend({"rgb": camera.data.output["rgb"][0]}, backend="numpy")["rgb"]
        try:
            from PIL import Image

            Image.fromarray(rgb[:, :, :3]).save(out_dir / "r1_standing.png")
            print(f"[INFO] Saved screenshot to {out_dir / 'r1_standing.png'}")
        except ImportError:
            np.save(out_dir / "r1_standing.npy", rgb)
            print(f"[WARN] PIL not available; saved raw array to {out_dir / 'r1_standing.npy'} instead.")
    else:
        print("[WARN] Camera produced no rgb output; screenshot not saved.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
