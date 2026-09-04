# R1 关节映射表：articulation 关节 -> unitree_hg 电机槽位

由 `deploy/tools/probe_cpp/gen_joint_map.py` 生成，2026-09-02。**不要手改。**

三个独立来源交叉核对：
1. `policy_interface.json` — 我们的关节顺序与动作索引
2. `unitree_sdk2/example/r1/low_level/r1_ankle_swing_example.cpp` 的
   `enum R1JointIndex` + motor-index 数组（enum 值是数组下标，**不是槽位号**）
3. `probe_lowstate --map` 的真机实测

⚠️ 来源 2 单独不算证据：同一文件里 `RightShoulderPitch = 29` 是笔误
（按数组位置应为 19）。当槽位号用会把右肩指令发到**头部**（槽 29 = head_pitch）。
bridge 必须按数组位置重建，不得 `#include` 这个 enum。

| art | joint | action | slot | 证据 | 实测 Δ |
|---|---|---|---|---|---|
| 0 | `left_hip_pitch_joint` | 0 | **0** | **实测确认** | -0.094 |
| 1 | `right_hip_pitch_joint` | 1 | **6** | **实测确认** | -0.145 |
| 2 | `waist_roll_joint` | 2 | **12** | 仅 enum | — |
| 3 | `left_hip_roll_joint` | 3 | **1** | 仅 enum | — |
| 4 | `right_hip_roll_joint` | 4 | **7** | 仅 enum | — |
| 5 | `waist_yaw_joint` | 5 | **13** | 仅 enum | — |
| 6 | `left_hip_yaw_joint` | 6 | **2** | 仅 enum | — |
| 7 | `right_hip_yaw_joint` | 7 | **8** | 仅 enum | — |
| 8 | `head_pitch_joint` | — | **29** | **实测确认** | +0.100 |
| 9 | `left_shoulder_pitch_joint` | 8 | **15** | 仅 enum | — |
| 10 | `right_shoulder_pitch_joint` | 9 | **22** | 仅 enum | — |
| 11 | `left_knee_joint` | 10 | **3** | **实测确认** | +0.107 |
| 12 | `right_knee_joint` | 11 | **9** | 仅 enum | — |
| 13 | `head_yaw_joint` | — | **30** | 仅 enum | — |
| 14 | `left_shoulder_roll_joint` | 12 | **16** | **实测确认** | -0.167 |
| 15 | `right_shoulder_roll_joint` | 13 | **23** | 仅 enum | — |
| 16 | `left_ankle_pitch_joint` | 14 | **4** | 仅 enum | — |
| 17 | `right_ankle_pitch_joint` | 15 | **10** | 仅 enum | — |
| 18 | `left_shoulder_yaw_joint` | 16 | **17** | 仅 enum | — |
| 19 | `right_shoulder_yaw_joint` | 17 | **24** | 仅 enum | — |
| 20 | `left_ankle_roll_joint` | 18 | **5** | **实测确认** | -0.218 |
| 21 | `right_ankle_roll_joint` | 19 | **11** | 仅 enum | — |
| 22 | `left_elbow_joint` | 20 | **18** | 仅 enum | — |
| 23 | `right_elbow_joint` | 21 | **25** | 仅 enum | — |
| 24 | `left_wrist_roll_joint` | 22 | **19** | 仅 enum | — |
| 25 | `right_wrist_roll_joint` | 23 | **26** | 仅 enum | — |

槽位覆盖: 26 个互不重复, 范围 0-30
实测确认: 6/26
