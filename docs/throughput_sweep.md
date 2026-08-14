# H1 flat-velocity throughput sweep on 2080Ti 11GB (RK-7)

Task: `Isaac-Velocity-Flat-H1-v0`, `--headless`, 40 iterations per level, `num_steps_per_env=24`
(from `rsl_rl_ppo_cfg.py`). This is the official-task throughput baseline required before
touching the R1 task, per the Week01 plan.

## RK-7 answer (the three numbers)

- **Max `num_envs` on this 2080Ti: 24576** (32768 hits `torch.OutOfMemoryError: CUDA out of
  memory`, confirmed from the actual crash log, not inferred).
- **Per-iteration wall clock**: grows with `num_envs`, from ~0.65s (num_envs=64) to ~3.2s
  (num_envs=24576) — see table. Pick the level that matches your training's `num_envs`.
- **Per-million-env-steps time**: drops from ~422s (num_envs=64) down to **~5.4s at the
  24576-env ceiling** — the practical throughput regime for training runs on this machine.

## Full sweep

| num_envs | result | peak GPU mem (MiB) | steady total_fps | per-iteration wall clock (s) | per-million-env-steps (s) |
|---|---|---|---|---|---|
| 64 | OK | 2700 | 2369 | 0.65 | 422.1 |
| 256 | OK | 2799 | 8830 | 0.70 | 113.3 |
| 1024 | OK | 3071 | 32861 | 0.75 | 30.4 |
| 2048 | OK | 3417 | 62408 | 0.79 | 16.0 |
| 4096 | OK | 4074 | 102491 | 0.96 | 9.8 |
| 8192 | OK | 5330 | 141713 | 1.39 | 7.1 |
| 16384 | OK | 7936 | 170021 | 2.31 | 5.9 |
| 24576 | OK | 10432 | 184190 | 3.20 | 5.4 |
| 32768 | **OOM** (`torch.OutOfMemoryError`, 10.56 GiB capacity, 10.23 GiB in use, tried to alloc 16 MiB more) | 10757 | - | - | - |

Notes:
- `total_fps` is aggregate environment-steps/sec across all parallel envs (rsl_rl's own metric),
  so per-million-env-steps time is directly comparable across `num_envs` levels.
- Throughput scales sub-linearly past ~8192 envs (diminishing fps/env), while GPU memory scales
  roughly linearly — memory is the binding constraint on this card, not compute.
- Raw per-level logs: `/tmp/sweep_<N>.log`; GPU memory samples: `/tmp/sweep_<N>_mem.log`.
- Reproduce with `./scripts/throughput_sweep.sh` (levels 64→16384) then
  `./scripts/throughput_sweep_extend.sh 24576 32768` (levels beyond that, appends to this file).
