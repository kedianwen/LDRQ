# reward_fix 简述

**要解决的问题**：修复 `first_train` 暴露的两腿岔开很宽、脚在地面蹭动的问题（`feet_air_time` 全程无梯度 + `joint_deviation_hip` 早早收敛到坏值）。

**相对上一版（first_train）的改动**：
- `feet_air_time` 权重 0.25 → 1.0（给它足够梯度去和"蹭地"这个更省力的局部最优抗衡）
- `feet_slide` 权重 -0.25 → -1.0（蹭地时这项太便宜了，加重惩罚）
- 拆分 `joint_deviation_hip` → `joint_deviation_hip_roll`(-0.4, 岔腿主因) / `joint_deviation_hip_yaw`(-0.1)
- 新增 `base_height_l2`(-1.0, target=0.72m)，防止蹲低变宽换稳定性
- 刻意不动 `dof_torques_l2`/`dof_acc_l2`/`action_rate_l2`——这三项本就偏爱低功耗的蹭地方案，调大只会帮倒忙

**关键参数组**：其余同 first_train（`num_envs=4096`, `dt=0.002`），本轮只训 800 iterations 做快速验证。

**结果**：`feet_air_time` 从死值 0.0045 涨到 0.19，`joint_deviation_hip_roll` 从 -0.19 降到 -0.012，reward 单调上升不再回落，蹭地问题解决。但回放发现新问题——左脚正常走、右脚靠点地骗 `feet_air_time` 奖励，见同级目录 `2026-08-15_10-56-44_reward_fix_1500`。
