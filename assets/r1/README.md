# assets/r1/

The R1 humanoid: source-of-truth URDF plus the Isaac Lab config generated from it.
26 joints (legs 2×6, waist 2, arms 2×5, head 2); the policy drives 24 of them (not the
head).

| Path | What it is | Tracked in git? |
|---|---|---|
| `R1.urdf` | Unitree's URDF for the R1 — the single source of truth for R1's kinematics/mass. Editing this is how you change the robot model. | yes |
| `meshes/` | STL collision/visual meshes referenced by `R1.urdf` via relative `meshes/*.STL` paths. | yes |
| `r1.py` | Isaac Lab `ArticulationCfg` (`R1_CFG`): spawn settings, default standing pose, per-joint-group PD gains. Imported by scripts as `from r1 import R1_CFG`. | yes |
| `usd/` | **Build artifact.** USD converted from `R1.urdf` by `../../scripts/convert_r1_urdf.py`. Not tracked in git — regenerate it rather than editing it, so the URDF stays authoritative and can't silently drift from a stale USD. | **no** (gitignored) |

## Where the URDF and meshes come from, and their license

`R1.urdf` and all of `meshes/` are **Unitree Robotics' files, unmodified**: byte-identical
to `robots/r1_description/` in [unitree_ros](https://github.com/unitreerobotics/unitree_ros)
(checked file by file against the upstream git hashes on 2026-10-01). They are under
Unitree's **BSD 3-Clause** license, reproduced in [`LICENSE`](LICENSE) in this folder,
not under the repository's Apache-2.0. `r1.py` and this README are this project's.

What this project changed is in `r1.py`, never in the URDF: the standing pose height,
and actuator limits and gains, which follow the URDF's own effort limits and Unitree's
`unitree_rl_mjlab` R1 config.

## Why USD is generated, not hand-authored

The simulated robot and the TSID/Pinocchio model must be the exact same robot
(same leg kinematics, foot frame, masses) for sim2real reasoning to hold. Since
the URDF is kept unmodified, deriving the USD from it mechanically guarantees that —
hand-editing a USD in Isaac Sim's GUI would not.

## Regenerating the USD

```bash
# from the repository root
~/IsaacLab/isaaclab.sh -p scripts/convert_r1_urdf.py
```

## Verifying the asset

```bash
~/IsaacLab/isaaclab.sh -p scripts/inspect_r1.py
```
Produces the joint/limit/mass table and a standing screenshot in `../../docs/`.

## History

Built in W01. In W04 the actuator limits and gains in `r1.py` were found to be stronger
than the real hardware's and were corrected to the URDF's limits and Unitree's own
values (see `r1.py`'s comments and [docs/project_history.md](../../docs/project_history.md)).
