# first_train 简述

**要解决的问题**：Week03 首次训练——验证 `flat_env_cfg.py` 的任务骨架（观测/动作/终止/奖励管理器）、非对称 AC（critic 观测组）能否让 R1 学会站立行走。无历史版本，本轮是 baseline。

**相对上一版的改动**：无（首训，直接用 H1 模板的标准权重移植过来，未针对 R1 调过）。

**关键参数组**：`num_envs=4096`, `dt=0.002`, `num_steps_per_env=24`, `max_iterations=1500`；奖励用 H1 模板默认权重（`feet_air_time=0.25`, `feet_slide=-0.25`, 单一 `joint_deviation_hip=-0.1` 未拆分, 无 `base_height_l2`）。

**结果**：reward -5.2→9.75，机器人不再摔倒，但 `scripts/play_r1.py` 回放显示两腿岔开很宽、脚在地面蹭动的退化步态；`feet_air_time` 项全程接近 0，从未产生有效梯度。诊断详见 `~/kdw/experiment_record/Week03_首训_宽步态与奖励回落问题.md`，催生了下一版 `reward_fix`。
