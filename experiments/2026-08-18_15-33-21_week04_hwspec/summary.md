# Run `2026-08-18_15-33-21_week04_hwspec`

- Task: `Isaac-Velocity-Flat-R1-v0`, `num_envs=4096`, `sim.dt=0.005`, `num_steps_per_env=24`
- Iterations completed: 2999/3000
- Git state at archive time: `313b658 (dirty -- uncommitted changes present at archive time)`

## Final metrics

| Metric | First logged | Last logged |
|---|---|---|
| Train/mean_reward | -1.511 (it 0) | 22.579 (it 2999) |
| Train/mean_episode_length | - | 1000.00 (it 2999) |
| Perf/total_fps | - | 55448 |

## Reward weights used

| Term | Weight |
|---|---|
| track_lin_vel_xy_exp | 1 |
| track_ang_vel_z_exp | 0.5 |
| alive | 0.15 |
| feet_swing_touchdown | 10 |
| lin_vel_z_l2 | -2 |
| ang_vel_xy_l2 | -0.05 |
| flat_orientation_l2 | -1 |
| base_height_l2 | -1 |
| dof_torques_l2 | -1e-05 |
| dof_acc_l2 | -2.5e-07 |
| action_rate_l2 | -0.01 |
| dof_pos_limits | -1 |
| feet_slide | -1 |
| feet_air_time_excess | -1 |
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
| Episode_Reward/action_rate_l2 | -0.0980 |
| Episode_Reward/alive | 0.1500 |
| Episode_Reward/ang_vel_xy_l2 | -0.0207 |
| Episode_Reward/base_height_l2 | -0.0002 |
| Episode_Reward/dof_acc_l2 | -0.0484 |
| Episode_Reward/dof_pos_limits | -0.0001 |
| Episode_Reward/dof_torques_l2 | -0.0143 |
| Episode_Reward/feet_air_time_excess | -0.0000 |
| Episode_Reward/feet_slide | -0.0443 |
| Episode_Reward/feet_swing_touchdown | 0.0708 |
| Episode_Reward/flat_orientation_l2 | -0.0025 |
| Episode_Reward/joint_deviation_arms | -0.0619 |
| Episode_Reward/joint_deviation_hip_roll | -0.0186 |
| Episode_Reward/joint_deviation_hip_yaw | -0.0118 |
| Episode_Reward/joint_deviation_waist | -0.0113 |
| Episode_Reward/lin_vel_z_l2 | -0.0090 |
| Episode_Reward/termination_penalty | 0.0000 |
| Episode_Reward/track_ang_vel_z_exp | 0.3177 |
| Episode_Reward/track_lin_vel_xy_exp | 0.9285 |
| Episode_Termination/bad_orientation | 0.0000 |
| Episode_Termination/base_height_low | 0.0000 |
| Episode_Termination/time_out | 3.7917 |
