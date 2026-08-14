# assets/r1/

The R1 humanoid: source-of-truth URDF plus the Isaac Lab config generated from it.

| Path | What it is | Tracked in git? |
|---|---|---|
| `R1.urdf` | Self-owned URDF — the single source of truth for R1's kinematics/mass. Editing this is how you change the robot model. | yes |
| `meshes/` | STL collision/visual meshes referenced by `R1.urdf` via relative `meshes/*.STL` paths. | yes |
| `r1.py` | Isaac Lab `ArticulationCfg` (`R1_CFG`): spawn settings, default standing pose, per-joint-group PD gains. Imported by scripts as `from r1 import R1_CFG`. | yes |
| `usd/` | **Build artifact.** USD converted from `R1.urdf` by `../../scripts/convert_r1_urdf.py`. Not tracked in git — regenerate it rather than editing it, so the URDF stays authoritative and can't silently drift from a stale USD. | **no** (gitignored) |

## Why USD is generated, not hand-authored

The simulated robot and the TSID/Pinocchio model must be the exact same robot
(same leg kinematics, foot frame, masses) for sim2real reasoning to hold. Since
R1's URDF is self-owned, deriving the USD from it mechanically guarantees that —
hand-editing a USD in Isaac Sim's GUI would not.

## Regenerating the USD

```bash
# from the R1process project root
~/IsaacLab/isaaclab.sh -p scripts/convert_r1_urdf.py
```

## Verifying the asset

```bash
~/IsaacLab/isaaclab.sh -p scripts/inspect_r1.py
```
Produces the joint/limit/mass table and a standing screenshot in `../../docs/`.
