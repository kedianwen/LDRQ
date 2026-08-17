# reward_fix_1500 简述

**要解决的问题**：不是奖励改动，是控制变量实验——排除"reward_fix 800 轮训练不够久"这个可能性，为左右脚不对称（右脚点地骗奖励）问题定性：是训练时长问题，还是 reward 函数结构性漏洞？

**相对上一版（reward_fix）的改动**：无配置改动，把同一份 `reward_fix` 配置从 800 轮延长跑到 1500 轮。

**关键参数组**：与 `reward_fix` 完全一致（`feet_air_time=1.0`, `feet_slide=-1.0`, `joint_deviation_hip_roll=-0.4` 等），仅 `max_iterations` 800→1500。

**结果**：`Train/mean_reward` 25.27→27.12 曲线健康，但 `feet_air_time`(0.190→0.215)、`feet_slide`(-0.025→-0.017)、`joint_deviation_hip_roll`(-0.012→-0.011) 三项从约第800迭代起基本走平，多训700轮没有实质变化；用户复核最终回放确认左右脚不对称依然存在。**结论：是 `feet_air_time_positive_biped` 本身的结构性漏洞（只要求瞬时单脚支撑，不要求真实完整迈步），不是欠训练**。方案见 `~/kdw/experiment_record/Week03_reward_fix_左右脚不对称蹭步问题与方案.md`，催生了下一版 `reward_fix_symmetry`（方案A）。
