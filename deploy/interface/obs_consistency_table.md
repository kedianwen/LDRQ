# W06 · 观测装配一致性对照表

> **本文件由 `deploy/tools/gen_obs_table.py` 生成，不要手改。**
> 生成器同时是**门禁**：四个文件里任意两个不一致就非零退出。
> 手抄的对照表只记录写它那天的信念；这个每次运行都重新核对。

训练来源：`2026-08-19_11-03-32_week04_nohead` / `Isaac-Velocity-Flat-R1-Play-v0`

交叉核对的四个文件（各自有独立的上机路径，所以能各自漂移）：

| # | 文件 | 它决定什么 |
|---|---|---|
| 1 | `deploy/interface/policy_interface.json` | 训练端真值，由 `dump_interface.py` 导出 |
| 2 | `r1_hw_bridge/include/r1_hw_bridge/joint_map.hpp` | bridge 从 35 槽里取哪 26 个、减哪些默认角 |
| 3 | `r1_policy_runner/config/policy_interface.yaml` | 策略节点的历史长度、动作缩放、默认角 |
| 4 | `r1_hw_bridge/config/bridge.yaml` | bridge 实际跑的频率 |

---

## 1 · 逐项对照

| 项 | 训练端 | 部署端实现 | 谁保证一致 |
|---|---|---|---|
| 控制周期 | 0.005 × 4 = 20 ms（50 Hz） | bridge `control_rate_hz`；命令线程独立跑 500 Hz | C1 三文件同值 + 节点自报速率 |
| 历史长度 | 5 帧 | `ObsAssembler`（在**策略节点**内，不在 bridge） | C2 |
| 展平顺序 | per-term (term-major), oldest frame first within each term | `ObsAssembler::Emit()` 外层遍历项、内层遍历帧 | C12 + `test_obs_assembler` |
| 帧长 / 总长 | 85 / 425 | 引擎输入维度不符则**拒绝启动** | C4 + 运行期 |
| 归一化 | none (empirical_normalization=false) -- feed raw values | 不做任何缩放，原值入网 | C12 |
| `base_ang_vel` (3) | 机体系角速度，rad/s | `LowState.imu_state.gyroscope`，直接透传 | — |
| `projected_gravity` (3) | `quat_rotate_inverse(q, (0,0,-1))` | `QuatRotateInverse()`，q 按 **(w,x,y,z)** 取 | 实测：直立时 ≈ (0,0,−1)，加速度计独立佐证 0.04 m/s² |
| `velocity_commands` (3) | 采样出的指令 | `~/cmd_vel`；**超时 500 ms 归零** | 遥控断了不能让机器人接着走 |
| `joint_pos` (26) | **相对默认姿态**，articulation 顺序 | `motor[kJointSlot[j]].q - kDefaultPos[j]` | C5/C6 + 阶段 3.4 |
| `joint_vel` (26) | **绝对值**，同顺序 | `motor[kJointSlot[j]].dq`，不减任何东西 | C5 |
| `actions` (24) | 上一周期策略原始输出（**缩放前**） | bridge 订阅 `~/action` 回填，不是 `joint_target` | C7/C8 |
| 动作映射 | `q_target[i] = default_pos[i] + scale[i] * action[i]` | 在**策略节点**内做，bridge 收到的已是目标角 | C9/C10 |

### 三个反复踩的坑

1. **展平是按项不是按帧。** 两种排法都是 425 个浮点数，错的那种照样加载、照样跑。
2. **`actions` 项喂的是缩放前的原始输出。** 训练时记录的就是网络输出本身；
   回填 `joint_target` 数值量级完全不同，且不报错。
3. **`action[i]` 和 `joint_pos[i]` 不是同一个关节。** 例：`action[8]` = `left_shoulder_pitch_joint`（落在 articulation 第 9 位），而 `joint_pos[8]` = `head_pitch_joint`。索引一样，关节不同。

---

## 2 · 26 关节：槽位 / 默认角 / 动作索引

| art | 关节 | hg 槽 | 默认角 | action | 证据 |
|---:|---|---:|---:|---:|---|
| 0 | `left_hip_pitch_joint` | 0 | -0.100 | 0 | **实测** |
| 1 | `right_hip_pitch_joint` | 6 | -0.100 | 1 | **实测** |
| 2 | `waist_roll_joint` | 12 | +0.000 | 2 | enum + 槽集 |
| 3 | `left_hip_roll_joint` | 1 | +0.000 | 3 | enum + 槽集 |
| 4 | `right_hip_roll_joint` | 7 | +0.000 | 4 | enum + 槽集 |
| 5 | `waist_yaw_joint` | 13 | +0.000 | 5 | enum + 槽集 |
| 6 | `left_hip_yaw_joint` | 2 | +0.000 | 6 | enum + 槽集 |
| 7 | `right_hip_yaw_joint` | 8 | +0.000 | 7 | enum + 槽集 |
| 8 | `head_pitch_joint` | 29 | +0.000 | — | **实测** |
| 9 | `left_shoulder_pitch_joint` | 15 | +0.350 | 8 | enum + 槽集 |
| 10 | `right_shoulder_pitch_joint` | 22 | +0.350 | 9 | enum + 槽集 |
| 11 | `left_knee_joint` | 3 | +0.300 | 10 | **实测** |
| 12 | `right_knee_joint` | 9 | +0.300 | 11 | enum + 槽集 |
| 13 | `head_yaw_joint` | 30 | +0.000 | — | enum + 槽集 |
| 14 | `left_shoulder_roll_joint` | 16 | +0.180 | 12 | **实测** |
| 15 | `right_shoulder_roll_joint` | 23 | -0.180 | 13 | enum + 槽集 |
| 16 | `left_ankle_pitch_joint` | 4 | -0.200 | 14 | enum + 槽集 |
| 17 | `right_ankle_pitch_joint` | 10 | -0.200 | 15 | enum + 槽集 |
| 18 | `left_shoulder_yaw_joint` | 17 | +0.000 | 16 | enum + 槽集 |
| 19 | `right_shoulder_yaw_joint` | 24 | +0.000 | 17 | enum + 槽集 |
| 20 | `left_ankle_roll_joint` | 5 | +0.000 | 18 | **实测** |
| 21 | `right_ankle_roll_joint` | 11 | +0.000 | 19 | enum + 槽集 |
| 22 | `left_elbow_joint` | 18 | +0.870 | 20 | enum + 槽集 |
| 23 | `right_elbow_joint` | 25 | +0.870 | 21 | enum + 槽集 |
| 24 | `left_wrist_roll_joint` | 19 | +0.000 | 22 | enum + 槽集 |
| 25 | `right_wrist_roll_joint` | 26 | +0.000 | 23 | enum + 槽集 |

槽集 = 普查实测 LIVE 集，6/26 条有直接掰动证据，其余由厂商 enum 加这条全局集合等式覆盖。

---

## 3 · 这张表**不**覆盖什么

门禁比对的是**静态常量**。以下三件事只有在机器人上跑起来才能验：

- bridge 有没有照着 `kJointSlot` 取数（表自洽 ≠ 代码照做）→ 上机清单阶段 3.3
- 默认角有没有真的减掉 → 阶段 3.4
- 四元数分量顺序 → 已实测定稿，见 W06 实验记录第 4 节

