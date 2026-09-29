# mission_ctl · 版本 B：用时间 / 角度 / 速度指挥机器人

不涉及 LLM。用桥**已有的** topic（`~/cmd_vel` 发、`~/imu` 和 `~/status` 收）把
"走 5 秒、左转 90 度、走 10 米"变成一条命令。**桥一行代码都不用改。**

```bash
python3 r1_mission_cli.py capability                      # 英文能力自述（LLM 的输入）+ JSON schema
python3 r1_mission_cli.py capability --json               # 同上，机器可读
python3 r1_mission_cli.py walk --seconds 5 --speed 0.4
python3 r1_mission_cli.py turn --deg 90 --left            # IMU 闭环
python3 r1_mission_cli.py run "walk 5s@0.4; turn left 90; walk 10m@0.4"
python3 r1_mission_cli.py run "..." --dry-run             # 仿真打印指令轨迹，不碰机器人
python3 r1_mission_cli.py json --file plan.json           # 以后 LLM 走这条
```

> **状态（2026-09-28）**：**已上真机**——机器人 Python 3.8.10 上 68/68；`stand 3s`、`walk 3s@0.2`
> 均 `DONE`；kp 扫描的 10 次记录都由它驱动。上机后按实测改了两处（见下文"首次上机"），
> 核心测试现为 88 项。尚未单独验收：转角的地面真值、Ctrl-C 在 foxy 下是否发出零（阶段 B）。

## 交互方案为什么是 Python，不是 bash / 不是 `ros2 param set`

按任务需要的**能力**来定，而不是按顺手程度：

| 方案 | 能持续发 cmd_vel 喂 deadman | 能读 `~/imu` 闭环转向 | 能在桥 DEGRADED 时中止 | 能串起多个原语 |
|---|---|---|---|---|
| `ros2 topic pub -r 10` | 能 | **不能** | **不能** | **不能**（停下来就得杀进程） |
| `ros2 param set` | 不能（参数不是指令通道） | 不能 | 不能 | 不能 |
| bash 脚本套上面两个 | 勉强 | **不能** | **不能** | 勉强 |
| **Python (rclpy) 节点** | 能 | 能 | 能 | 能 |

后两列是硬需求：**转向要闭环就必须订阅 `~/imu`**，而 `ros2 topic pub` 只会发不会收。
所以执行体必须是一个节点。**但操作界面做成 CLI**——现场效率就是一行命令、不改文件：

- 一次性动作：`walk --seconds 5`
- 可复现序列：`run "walk 5s@0.4; turn left 90; walk 10m"` 或 `run --file demo.mission`
- 上机之前：同一条命令加 `--dry-run`，在开发机上把整条轨迹打出来

而且 CLI 的脚本串和以后 LLM 的 JSON **编译到同一个 Prim 列表**（`plan.compile_plan`），
所以 LLM 层只是换一个前端，执行器和全部校验完全复用。

## 一次调用 = 一个进程 = 一个计划（没有常驻守护进程）

这是安全属性，不是简化：桥的 `cmd_vel_timeout_ms=500` 会把过期指令衰减到零，
所以 mission 进程**退出、崩溃、被 kill，机器人都会停**——不依赖谁记得去停它。
节点还会在启动时数 `~/cmd_vel` 上的发布者，发现已有别人在发就拒绝当第二写者
（两个调度器抢同一个话题，从机器人这侧看和"策略压不住指令"长得一模一样）。

## 两个原语的不对称（这是本目录的核心）

| | 能否闭环 | 依据 | 精度 |
|---|---|---|---|
| `turn <deg>` | **能** | 桥已在 `~/imu` 以 50 Hz 发 `[qw,qx,qy,qz,...]`；`yaw=atan2(2(wz+xy), 1−2(y²+z²))`，解缠累加 | 控制器**瞄准目标角**而不是瞄容差带（否则系统性欠转到 87°）。10 Hz 下残差 <1°，容差 3° 只用于验收告警 |
| `walk <m>` | **不能** | 425 维观测里没有 base 线速度，也没有任何里程计话题 | 只能 `时间 = 距离 / 速度`，速度见下 |

`mission.yaml` 的 `v_cal` 有三种取值，含义各不相同：

| `v_cal:` | 按米走 | 距离怎么上报 |
|---|---|---|
| **`commanded`（当前默认，2026-09-28 决定）** | 允许，按**指令速度**换算时间 | `~10 m if speed = command; not measured`，**不带误差棒**——没测过就不编一个 |
| `0.35`（实测数） | 允许，按实测速度换算 | `10 ± 1.5 m`（`v_cal_rel_err`） |
| `unmeasured` / 缺省 | **拒绝** | — |

改成 `commanded` 的理由：没有场地做测速，而这个任务宏观上只要"几米"量级。
**缺省仍然是拒绝**：配置必须把"假设"写出来，否则"10 米"会变成一个看起来像测量值的数字。
能力自述会同步告诉 LLM "my real walking speed has not been measured"。
以后若测了速度，把数字填进去即可，其余不变（`deploy/tools/probe_cpp/vcal.py` 留着备用）。

## 训练分布决定的两条限制

- **`min_primitive_s: 2.0`** —— 训练时速度指令每 10 s 才阶跃一次
  （`resampling_time_range=(10.0, 10.0)`）。阶跃本身在分布内，但一串半秒的原语
  连起来是策略很少见过的瞬态。2 秒是有理由的下限，不是随手定的。
- **`ramp_s: 0.5`** —— 同理，斜坡不要钱，把每个原语最开始几个控制周期挪出最陡的瞬态。
- 另外记一笔：站立在训练里只占 `rel_standing_envs=0.02`，`stand` 是弱训练状态，
  要测它能站多久，不要假设。

## 校验与预算：整份计划要么全过要么全拒

`compile_plan` 在**发出第一条 cmd_vel 之前**做完所有拒绝：包线（`vx∈[0,1]`、`wz∈[-0.5,0.5]`、
`vy` 钉死轴**拒绝而非截断**）、原语时长下限、总时长 / 总距离 / 原语条数预算。
包线本身从 `$R1_DEPLOY_ROOT/interface/command_envelope.json`（policy_pack 装包时落地的）读，
读不到再退到桥的 `bridge.yaml`——**和桥 clamp 的是同一组数字**，
所以 `capability` 打印的那句 "I cannot move sideways" 不可能和真正执行的包线走偏。

## 测试

```bash
python3 tests/test_core.py      # 88 项，无需 ROS / 无需 pytest / 无需机器人
```

不依赖 pytest 是有意的：机器人上 apt 已损坏、Python 是 3.8，
一套在真正要紧的那台机器上跑不起来的测试不算测试。

**联调替身桥又抓到三个单元测试看不见的 bug**（单元测试永远传入 `RUNNING`，碰不到 ROS 这一侧）：
- `~/imu` 订阅用了默认的 RELIABLE，而桥发的是 BEST_EFFORT——两者**不会连接**，
  真机上每一次闭环转向都会以 "no IMU yaw" 中止
- 节点以 `UNKNOWN` 起步、0.1 s 就发第一拍，而执行器把非 RUNNING 一律当中止——
  除非状态消息恰好 0.1 s 内到达，**每次真机运行都会立即中止**；且状态永远不来时会无限等待
- humble 起 rclpy 在 SIGINT 时关掉 context，"退出前发零"静默失败，
  Ctrl-C 后桥收到的最后一条仍是 `vx=0.30`，只能靠 500 ms deadman 停

现在：订阅按桥的 QoS 匹配；无状态 = 等待，10 s 超时并说出原因；运行中状态静默 3 s 即中止；
自己接管 SIGINT，Ctrl-C 后最后一条实测为零（foxy 的信号处理不同，上机要再看一次）。

能力自述（`capability`）**默认英文**，因为它就是 LLM 的输入；`describe("zh")` 保留给操作者。

**首次上机（2026-09-28，kp 扫描，全程松吊装）又暴露两处：**
- **转向超时的基数算错了。** 超时原来是 `2 × 角度/角速度`，但执行器在最后 25° 会减速
  （下限 0.35 倍），180° @0.4 rad/s 实际要 9.25 s 而不是 7.85 s，所谓 "2 倍" 其实只有 1.7 倍。
  10 次记录里 4 次在第一个转弯超时（只转到 108°–177°）。现在超时和计划总时长都用
  `plan.expected_turn_s()`（含斜坡和减速段，和执行器逐周期仿真误差 <0.1 s）。
- **吊装下的转向不代表机器人的转向。** 完成的转弯是理想时长的 1.1–1.4 倍；第一个 180° 之后，
  直走 5 s 会往回偏 6°–61°，kp=0.9 的第二个左转甚至往右转了 48°——这是吊绳扭转，
  不是增益（同一个 kp=1.10 重复两次，一次转完、一次超时）。转角精度的验收放在阶段 B 无吊装做。

这套测试在编写过程中抓到四个真 bug，都是只看代码不容易发现的：
原语切换时 `Output` 少传一个参数（**每个多原语计划必崩**）、
settle 计时每周期被重置（**计划永远到不了 DONE**）、
settle 期间反复改写最后一条原语的 `actual_s`（时长多报 1 s）、
以及 `stop` 没有时长导致**以 stop 结尾的计划永不结束**。

## 上机前

桥必须在 `RUNNING`（执行器不会在 `WAITING_POLICY` 时开始跑第一条原语）；
手柄必须在**开发者模式**，否则厂家运控在 500 Hz 抢 `rt/lowcmd`；
`enable_output:=true` 之前先用 `--dry-run`，再用 `--yes` 之外的交互确认走一遍。
