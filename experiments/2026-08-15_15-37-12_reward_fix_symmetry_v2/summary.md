# Run `2026-08-15_15-37-12_reward_fix_symmetry_v2`

- Task: `Isaac-Velocity-Flat-R1-v0`, `num_envs=4096`, `sim.dt=0.002`, `num_steps_per_env=24`
- Iterations completed: 1499/1500
- Git state at archive time: `e18757d (dirty -- uncommitted changes present at archive time)`

## Final metrics

| Metric | First logged | Last logged |
|---|---|---|
| Train/mean_reward | -5.343 (it 0) | 23.113 (it 1499) |
| Train/mean_episode_length | - | 1000.00 (it 1499) |
| Perf/total_fps | - | 37780 |

## Reward weights used

| Term | Weight |
|---|---|
| track_lin_vel_xy_exp | 1 |
| track_ang_vel_z_exp | 0.5 |
| alive | 0.15 |
| feet_air_time | 1 |
| lin_vel_z_l2 | -2 |
| ang_vel_xy_l2 | -0.05 |
| flat_orientation_l2 | -1 |
| base_height_l2 | -1 |
| dof_torques_l2 | -1e-05 |
| dof_acc_l2 | -2.5e-07 |
| action_rate_l2 | -0.01 |
| dof_pos_limits | -1 |
| feet_slide | -1 |
| feet_air_time_symmetry | -0.3 |
| joint_deviation_arms | -0.1 |
| joint_deviation_waist | -0.1 |
| joint_deviation_hip_roll | -0.4 |
| joint_deviation_hip_yaw | -0.1 |
| termination_penalty | -200 |

## Command ranges used

| Command | Range |
|---|---|
| lin_vel_x | [0, 0.5] |
| lin_vel_y | [0, 0] |
| ang_vel_z | [0, 0] |
| heading | [-3.14159, 3.14159] |

## Final per-term breakdown (last logged value)

| Term | Value |
|---|---|
| Episode_Reward/action_rate_l2 | -0.1254 |
| Episode_Reward/alive | 0.1500 |
| Episode_Reward/ang_vel_xy_l2 | -0.0094 |
| Episode_Reward/base_height_l2 | -0.0003 |
| Episode_Reward/dof_acc_l2 | -0.0154 |
| Episode_Reward/dof_pos_limits | -0.0012 |
| Episode_Reward/dof_torques_l2 | -0.1290 |
| Episode_Reward/feet_air_time | 0.1909 |
| Episode_Reward/feet_air_time_symmetry | -0.0599 |
| Episode_Reward/feet_slide | -0.0245 |
| Episode_Reward/flat_orientation_l2 | -0.0132 |
| Episode_Reward/joint_deviation_arms | -0.0673 |
| Episode_Reward/joint_deviation_hip_roll | -0.0142 |
| Episode_Reward/joint_deviation_hip_yaw | -0.0086 |
| Episode_Reward/joint_deviation_waist | -0.0081 |
| Episode_Reward/lin_vel_z_l2 | -0.0124 |
| Episode_Reward/termination_penalty | 0.0000 |
| Episode_Reward/track_ang_vel_z_exp | 0.4044 |
| Episode_Reward/track_lin_vel_xy_exp | 0.9019 |
| Episode_Termination/bad_orientation | 0.0000 |
| Episode_Termination/base_height_low | 0.0000 |
| Episode_Termination/time_out | 4.4167 |
