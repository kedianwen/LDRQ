#!/usr/bin/env python3
"""Convert ``assets/r1/R1.urdf`` to a USD so the sim robot == the Pinocchio model.

For R1 we own the URDF, so we generate the simulation USD directly from it:
the simulated robot and the Pinocchio model are then identical (same leg kinematics, foot frame, and masses).

Run once inside the Isaac runtime, from the project root::

    ~/IsaacLab/isaaclab.sh -p scripts/convert_r1_urdf.py

Output: ``assets/r1/usd/R1.usd`` (consumed by ``r1.R1_CFG``, see ``assets/r1/r1.py``).

Prerequisite: the meshes referenced by R1.urdf must exist at ``assets/r1/meshes/``
(the URDF uses relative ``meshes/*.STL`` paths). Conversion fails without them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "r1"
_URDF = _ASSET_DIR / "R1.urdf"
_USD_DIR = _ASSET_DIR / "usd"
_USD_NAME = "R1.usd"

parser = argparse.ArgumentParser(description="Convert R1.urdf to USD.")
parser.add_argument("--urdf", type=str, default=str(_URDF), help="Input URDF path.")
parser.add_argument("--usd-dir", type=str, default=str(_USD_DIR), help="Output USD directory.")
parser.add_argument("--usd-name", type=str, default=_USD_NAME, help="Output USD file name.")
parser.add_argument("--fix-base", action="store_true", help="Fix the base (default: floating).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Conversion is offscreen; force headless regardless of CLI.
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----- after the app is up -----
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402


def main() -> None:
    urdf_path = Path(args_cli.urdf).resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    meshes_dir = urdf_path.parent / "meshes"
    if not meshes_dir.is_dir():
        print(
            f"[WARN] {meshes_dir} not found. R1.urdf references 'meshes/*.STL' relative to the "
            "URDF; conversion will fail without them. Place the R1 mesh folder there first.",
            flush=True,
        )

    usd_dir = Path(args_cli.usd_dir).resolve()
    usd_dir.mkdir(parents=True, exist_ok=True)

    # merge_fixed_joints lumps the fixed parallel-ankle rods into their parent,
    # matching how Pinocchio treats them. fix_base=False -> free-flyer base.
    # joint_drive here only sets the USD's native drive; the WBC env overrides
    # actuator gains via R1_CFG (zeroing legs for pure TSID effort control).
    cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(usd_dir),
        usd_file_name=str(args_cli.usd_name),
        fix_base=bool(args_cli.fix_base),
        merge_fixed_joints=True,
        force_usd_conversion=True,
        make_instanceable=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=10.0),
        ),
    )

    converter = UrdfConverter(cfg)
    out_path = converter.usd_path
    print(f"[INFO] Converted URDF -> USD: {out_path}", flush=True)

    # Report the articulation joints so the user can confirm the 26 revolute DOF
    # are present with the same names as the URDF (TSID joint mapping relies on this).
    try:
        import isaacsim.core.utils.stage as stage_utils  # noqa: E402
        from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
        from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402

        stage_utils.create_new_stage()
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
        robot = Articulation(
            ArticulationCfg(
                prim_path="/World/R1",
                spawn=sim_utils.UsdFileCfg(usd_path=out_path),
                actuators={
                    # ArticulationCfg requires an actuator mapping. This follows
                    # Isaac Lab examples/tests, which use a catch-all implicit
                    # actuator when loading an articulation only to inspect it.
                    "all_joints": ImplicitActuatorCfg(
                        joint_names_expr=[".*"],
                        stiffness=100.0,
                        damping=10.0,
                    ),
                },
            )
        )
        sim.reset()
        names = list(robot.data.joint_names)
        print(f"[INFO] USD articulation has {len(names)} joints:", flush=True)
        for n in names:
            print(f"    {n}", flush=True)
    except Exception as exc:  # introspection is best-effort
        print(f"[WARN] Could not introspect converted USD joints: {exc}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
