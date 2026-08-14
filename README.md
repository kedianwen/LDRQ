# R1process

Working directory for the R1 Sim2Real project's Isaac Lab side: R1 asset pipeline,
the R1 manager-based task, per-week verification scripts, and their output
reports. This is the practical counterpart to the
[12-week dev plan](~/kdw/development%20plan/) — currently at **Week02 complete**
(see `docs/`).

## Layout

- `assets/` — the R1 robot definition: URDF, meshes, Isaac Lab `ArticulationCfg`,
  and the USD build artifact. See `assets/README.md`.
- `tasks/` — the R1 flat-ground velocity-tracking gym task (manager-based:
  scene/actions/observations/rewards/terminations/events). See `tasks/README.md`.
- `scripts/` — tools that operate on the asset and task: URDF→USD conversion,
  load/limit/standing verification, task smoke test, GPU throughput sweeps.
  See `scripts/README.md`.
- `docs/` — dated verification reports and screenshots produced by the scripts
  above (Week01/Week02 exit-criteria evidence). See `docs/README.md`.

## Environment

Isaac Lab 2.1.0 + Isaac Sim 4.5.0 + torch 2.5.1+cu118, conda env `env_isaaclab`.
All scripts run through `~/IsaacLab/isaaclab.sh -p ...` from this directory (the
project root) — see each script's docstring/header for the exact invocation.

## Hardware target

R1 is the **EDU version** with an onboard 8-core Jetson Orin NX (40-100 TOPS) —
no external compute board needed for on-robot deployment (Week05+).
