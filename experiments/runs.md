# Training run index

One row per archived run (see `README.md` for what "archived" means and how
to add a row). Newest first. `run_id` links to that run's folder here for
the full config + summary; the same `run_id` names the matching directory
under `logs/rsl_rl/r1_flat/` if you need the raw checkpoints/tensorboard
event files (git-ignored, local only).

<!-- archive_run.py inserts rows below this line -->

| run_id | scale | progress | train reward trend | final ep length | details |
|---|---|---|---|---|---|
| 2026-08-14_14-22-09_first_train | num_envs=4096 | it 1499/1500 | reward -5.20 -> 9.75 | eplen 915.1 | [2026-08-14_14-22-09_first_train](2026-08-14_14-22-09_first_train/summary.md) |
| 2026-08-14_16-19-36_reward_fix | num_envs=4096 | it 799/800 | reward -5.33 -> 25.27 | eplen 1000.0 | [2026-08-14_16-19-36_reward_fix](2026-08-14_16-19-36_reward_fix/summary.md) |
| 2026-08-15_10-56-44_reward_fix_1500 | num_envs=4096 | it 1499/1500 | reward -5.33 -> 27.12 | eplen 1000.0 | [2026-08-15_10-56-44_reward_fix_1500](2026-08-15_10-56-44_reward_fix_1500/summary.md) |
| 2026-08-15_13-25-00_reward_fix_symmetry | num_envs=4096 | it 1499/1500 | reward -5.38 -> 24.90 | eplen 995.0 | [2026-08-15_13-25-00_reward_fix_symmetry](2026-08-15_13-25-00_reward_fix_symmetry/summary.md) |
| 2026-08-15_15-37-12_reward_fix_symmetry_v2 | num_envs=4096 | it 1499/1500 | reward -5.34 -> 23.11 | eplen 1000.0 | [2026-08-15_15-37-12_reward_fix_symmetry_v2](2026-08-15_15-37-12_reward_fix_symmetry_v2/summary.md) |
| 2026-08-15_17-34-57_touchdown_gate | num_envs=4096 | it 1499/1500 | reward -1.06 -> 28.63 | eplen 998.1 | [2026-08-15_17-34-57_touchdown_gate](2026-08-15_17-34-57_touchdown_gate/summary.md) |
