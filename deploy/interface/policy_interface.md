# R1 policy deployment interface

Generated from `2026-08-19_11-03-32_week04_nohead` by `deploy/tools/dump_interface.py`.

## Control loop

- sim dt `0.005` x decimation `4` = **20.0 ms** per control step (**50.0 Hz**)

## Observation (policy input)

**425 floats** = 5 frames x 85 per frame, flattened **per term** (term-major), oldest frame first inside each term.

| term | width/frame | x history | offset |
|---|---|---|---|
| `base_ang_vel` | 3 | 15 | 0 |
| `projected_gravity` | 3 | 15 | 15 |
| `velocity_commands` | 3 | 15 | 30 |
| `joint_pos` | 26 | 130 | 45 |
| `joint_vel` | 26 | 130 | 175 |
| `actions` | 24 | 120 | 305 |

No input normalisation: the policy was trained with `empirical_normalization=false`, so the exported graph is the whole computation. Feed raw values.

## Action (policy output)

**24 floats**, articulation order, mapped as `q_target[i] = default_pos[i] + scale[i] * action[i]`.

| # | joint | scale | default (rad) |
|---|---|---|---|
| 0 | `left_hip_pitch_joint` | 0.1500 | -0.1000 |
| 1 | `right_hip_pitch_joint` | 0.1500 | -0.1000 |
| 2 | `waist_roll_joint` | 0.1500 | 0.0000 |
| 3 | `left_hip_roll_joint` | 0.1500 | 0.0000 |
| 4 | `right_hip_roll_joint` | 0.1500 | 0.0000 |
| 5 | `waist_yaw_joint` | 0.1500 | 0.0000 |
| 6 | `left_hip_yaw_joint` | 0.1500 | 0.0000 |
| 7 | `right_hip_yaw_joint` | 0.1500 | 0.0000 |
| 8 | `left_shoulder_pitch_joint` | 0.3750 | 0.3500 |
| 9 | `right_shoulder_pitch_joint` | 0.3750 | 0.3500 |
| 10 | `left_knee_joint` | 0.1500 | 0.3000 |
| 11 | `right_knee_joint` | 0.1500 | 0.3000 |
| 12 | `left_shoulder_roll_joint` | 0.3750 | 0.1800 |
| 13 | `right_shoulder_roll_joint` | 0.3750 | -0.1800 |
| 14 | `left_ankle_pitch_joint` | 0.3125 | -0.2000 |
| 15 | `right_ankle_pitch_joint` | 0.3125 | -0.2000 |
| 16 | `left_shoulder_yaw_joint` | 0.4125 | 0.0000 |
| 17 | `right_shoulder_yaw_joint` | 0.4125 | 0.0000 |
| 18 | `left_ankle_roll_joint` | 0.3125 | 0.0000 |
| 19 | `right_ankle_roll_joint` | 0.3125 | 0.0000 |
| 20 | `left_elbow_joint` | 0.4125 | 0.8700 |
| 21 | `right_elbow_joint` | 0.4125 | 0.8700 |
| 22 | `left_wrist_roll_joint` | 0.4125 | 0.0000 |
| 23 | `right_wrist_roll_joint` | 0.4125 | 0.0000 |

## Articulation joint order (26 joints)

The `joint_pos` / `joint_vel` observation terms cover **all** joints in this order -- including the two head joints the policy does not drive. `joint_pos` is relative to the default pose; `joint_vel` is absolute.

| # | joint | default (rad) | policy-actuated |
|---|---|---|---|
| 0 | `left_hip_pitch_joint` | -0.1000 | yes |
| 1 | `right_hip_pitch_joint` | -0.1000 | yes |
| 2 | `waist_roll_joint` | 0.0000 | yes |
| 3 | `left_hip_roll_joint` | 0.0000 | yes |
| 4 | `right_hip_roll_joint` | 0.0000 | yes |
| 5 | `waist_yaw_joint` | 0.0000 | yes |
| 6 | `left_hip_yaw_joint` | 0.0000 | yes |
| 7 | `right_hip_yaw_joint` | 0.0000 | yes |
| 8 | `head_pitch_joint` | 0.0000 | NO |
| 9 | `left_shoulder_pitch_joint` | 0.3500 | yes |
| 10 | `right_shoulder_pitch_joint` | 0.3500 | yes |
| 11 | `left_knee_joint` | 0.3000 | yes |
| 12 | `right_knee_joint` | 0.3000 | yes |
| 13 | `head_yaw_joint` | 0.0000 | NO |
| 14 | `left_shoulder_roll_joint` | 0.1800 | yes |
| 15 | `right_shoulder_roll_joint` | -0.1800 | yes |
| 16 | `left_ankle_pitch_joint` | -0.2000 | yes |
| 17 | `right_ankle_pitch_joint` | -0.2000 | yes |
| 18 | `left_shoulder_yaw_joint` | 0.0000 | yes |
| 19 | `right_shoulder_yaw_joint` | 0.0000 | yes |
| 20 | `left_ankle_roll_joint` | 0.0000 | yes |
| 21 | `right_ankle_roll_joint` | 0.0000 | yes |
| 22 | `left_elbow_joint` | 0.8700 | yes |
| 23 | `right_elbow_joint` | 0.8700 | yes |
| 24 | `left_wrist_roll_joint` | 0.0000 | yes |
| 25 | `right_wrist_roll_joint` | 0.0000 | yes |
