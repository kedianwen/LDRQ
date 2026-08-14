# R1 Joint / Limit / Drive Verification (Week01 exit criterion)

- Total DOF: **26** (expected 26 actuated)
- Base standing height before/after 700 settle steps: 0.722 m -> 0.719 m (drop 3.1 mm)
- Max roll/pitch after settle: 0.09 deg
- **Standing stable: YES**

| # | joint | lower (rad) | upper (rad) | default pos (rad) | effort limit (N·m) | stiffness | damping |
|---|---|---|---|---|---|---|---|
| 0 | left_hip_pitch_joint | -2.932 | 2.548 | -0.260 | 150.0 | 1200.0 | 100.0 |
| 1 | right_hip_pitch_joint | -2.932 | 2.548 | -0.260 | 150.0 | 1200.0 | 100.0 |
| 2 | waist_roll_joint | -0.524 | 0.524 | 0.000 | 60.0 | 400.0 | 20.0 |
| 3 | left_hip_roll_joint | -1.047 | 1.745 | 0.000 | 150.0 | 1200.0 | 100.0 |
| 4 | right_hip_roll_joint | -1.745 | 1.047 | 0.000 | 150.0 | 1200.0 | 100.0 |
| 5 | waist_yaw_joint | -2.618 | 2.618 | 0.000 | 60.0 | 400.0 | 20.0 |
| 6 | left_hip_yaw_joint | -2.740 | 2.740 | 0.000 | 150.0 | 1200.0 | 100.0 |
| 7 | right_hip_yaw_joint | -2.740 | 2.740 | 0.000 | 150.0 | 1200.0 | 100.0 |
| 8 | head_pitch_joint | -0.628 | 0.628 | 0.000 | 33.0 | 40.0 | 10.0 |
| 9 | left_shoulder_pitch_joint | -3.142 | 2.094 | 0.150 | 40.0 | 80.0 | 15.0 |
| 10 | right_shoulder_pitch_joint | -3.142 | 2.094 | 0.150 | 40.0 | 80.0 | 15.0 |
| 11 | left_knee_joint | -0.175 | 2.426 | 0.520 | 150.0 | 1200.0 | 100.0 |
| 12 | right_knee_joint | -0.175 | 2.426 | 0.520 | 150.0 | 1200.0 | 100.0 |
| 13 | head_yaw_joint | -2.007 | 2.007 | 0.000 | 33.0 | 40.0 | 10.0 |
| 14 | left_shoulder_roll_joint | -0.227 | 2.478 | 0.100 | 40.0 | 80.0 | 15.0 |
| 15 | right_shoulder_roll_joint | -2.478 | 0.227 | -0.100 | 40.0 | 80.0 | 15.0 |
| 16 | left_ankle_pitch_joint | -0.873 | 0.576 | -0.260 | 150.0 | 1200.0 | 150.0 |
| 17 | right_ankle_pitch_joint | -0.873 | 0.576 | -0.260 | 150.0 | 1200.0 | 150.0 |
| 18 | left_shoulder_yaw_joint | -1.920 | 1.920 | 0.000 | 40.0 | 80.0 | 15.0 |
| 19 | right_shoulder_yaw_joint | -1.920 | 1.920 | 0.000 | 40.0 | 80.0 | 15.0 |
| 20 | left_ankle_roll_joint | -0.262 | 0.262 | 0.000 | 150.0 | 1200.0 | 150.0 |
| 21 | right_ankle_roll_joint | -0.262 | 0.262 | 0.000 | 150.0 | 1200.0 | 150.0 |
| 22 | left_elbow_joint | -0.976 | 2.185 | 0.350 | 40.0 | 80.0 | 15.0 |
| 23 | right_elbow_joint | -0.976 | 2.185 | 0.350 | 40.0 | 80.0 | 15.0 |
| 24 | left_wrist_roll_joint | -1.920 | 1.920 | 0.000 | 40.0 | 80.0 | 15.0 |
| 25 | right_wrist_roll_joint | -1.920 | 1.920 | 0.000 | 40.0 | 80.0 | 15.0 |

## Mass sanity (27 bodies, total 28.84 kg)

- Bodies with zero/negative mass: **none**

## Joint limit sanity

- Joints with degenerate (zero-width) position limits: **none**
