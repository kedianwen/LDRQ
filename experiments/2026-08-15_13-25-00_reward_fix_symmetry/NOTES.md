# reward_fix_symmetry 简述

**要解决的问题**：`reward_fix`/`reward_fix_1500` 暴露的左右脚不对称"点地骗奖励"问题——右脚靠快速点地满足单脚支撑条件骗取 `feet_air_time` 奖励。按 `~/kdw/experiment_record/Week03_reward_fix_左右脚不对称蹭步问题与方案.md` 的方案A执行。

**相对上一版（reward_fix_1500）的改动**：新增 `feet_air_time_symmetry`（`tasks/r1_flat/mdp.py`，权重 -0.3），用 `current_air_time`/`current_contact_time` 算每只脚的瞬时"占空比"，惩罚左右差值。其余权重不变。

**关键参数组**：`num_envs=4096`, `dt=0.002`, `max_iterations=1500`；`feet_air_time_symmetry` weight=-0.3。

**结果：方案A未达预期，且引入了新的退化步态，本轮判定失败，不建议采用。**
- `Episode_Reward/feet_air_time` 从 reward_fix_1500 的 0.215 崩到 0.0008，全程被压制。
- 回放（`videos/play/rl-video-step-0.mp4`）显示：两脚并拢、屈膝，呈同步小跳/蹭步姿态，不是正常的左右交替迈步。
- 根因（事后定位）：`current_air_time`/`current_contact_time` 是"当前这一阶段已持续多久"的瞬时值,不是一段时间窗口内的占空比。单脚支撑的瞬间,摆动脚的 duty≈1、支撑脚的 duty≈0,差值天然≈1——这个惩罚几乎和"任意一次单脚支撑"本身一样重,而不是专门惩罚"不对称的点地"。于是策略学会的是干脆避免单脚支撑（双脚同步挪动/小跳),而不是让两脚步幅对称。这是这次实现本身的公式设计问题，与最初诊断的漏洞（`feet_air_time_positive_biped` 只看瞬时单脚支撑）不冲突,是在修复它时引入的新 bug。

**下一步（提议，未执行）**：改用 `last_air_time`（上一次完整摆动相的持续时长，而非当前进行中的瞬时值）比较左右脚,例如 `|last_air_time_left - last_air_time_right|`——只在每次摆动相真正结束时更新,不会在单脚支撑进行中就产生虚假的高惩罚。详见待写的后续记录。
