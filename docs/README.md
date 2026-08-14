# docs/

Verification reports and screenshots produced by `../scripts/`. These are
results/evidence (Week01/Week02 exit-criteria deliverables), not build
artifacts, so they're tracked in git — re-running the scripts overwrites them
with fresh numbers, which is the point.

| File | Produced by | Contents |
|---|---|---|
| `joint_check.md` | `scripts/inspect_r1.py` | Per-joint position limits, effort limits, PD gains, default pose; body mass sanity check (Week01); standing-stability check after a ~11s PD-hold settle (Week02: **stable**, 3.1mm height drop, 0.19° max tilt — see `assets/r1/r1.py` for the gain/timestep tuning that got here). |
| `r1_standing.png` | `scripts/inspect_r1.py` | Screenshot of R1 standing after the settle period. |
| `throughput_sweep.md` | `scripts/throughput_sweep.sh` + `throughput_sweep_extend.sh` | RK-7 answer: max `num_envs` on this machine's 2080Ti before OOM, steady-state fps, and per-million-env-step time at each level. |
