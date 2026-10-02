# R1 端侧栈 v1.1.0：在机器人上用英文指令操控宇树 R1

这个压缩包是**跑在机器人上**的全部内容，适用于宇树 R1 EDU（板载 Jetson Orin NX）。装好后可以：

- 编译并安装行走策略，并对照参考数值做校验；
- 起控制栈；
- 用一句英文指令操控机器人，例如：
  `python3 $M ask "Walk forward for 3 seconds at 0.2 meters per second."`

机器人自己 GPU 上的一个小语言模型先把这句话转写成计划；再由确定性代码按策略训练时的范围检查计划，并打印给你看；**你按 `y` 之后机器人才会动**。机器人端不需要联网。

源码、设计说明和实验结果见 <https://github.com/kedianwen/LDRQ>。本版本由其中的 `v1.1.0` 标签构建。

> **动之前先读"安全"一节。** 这是研究用代码，机器人会摔倒，也可能伤人。头几次务必吊装运行，急停遥控拿在手上。

## 发布内容

| 附件 | 大小 | 内容 |
|---|---|---|
| `release_v1.1.0.tar.gz` | 约 0.5 GB | 本说明、代码、策略包、JetPack 5 版 Ollama 模型服务 |
| `qwen3-1.7b_r1-robot-v1.1.0.tar.gz` | 约 1.3 GB | 语言模型 qwen3:1.7b（Q4_K_M，Ollama 格式） |
| `SHA256SUMS` | — | 两个压缩包的校验值 |

两个压缩包都解到同一个目录 `r1-robot-v1.1.0/`：

```
r1-robot-v1.1.0/
  README.md, README_zh.md   英文 / 中文说明
  selftest.sh               不需要机器人运动的自检
  deploy/                   C++/ROS 2 硬件桥与 TensorRT 策略节点（在机器人上编译）
  bundles/<run_id>/         训练好的策略：ONNX、接口、增益、指令包线、parity 参考数据
  policy_pack/              装策略包：生成配置、编译、构建引擎、parity 闸门
  mission_ctl/              指令层：run / ask / capability
  llm/                      ollama_ctl.sh、Ollama 运行时（llm/ollama/）、模型（llm/models/）
  LICENSE, NOTICE, THIRD_PARTY_NOTICES.md
```

策略是 `2026-08-19_11-03-32_week04_nohead`：425 维本体感知输入 → 24 个关节目标，50 Hz，在 Isaac Lab 中训练。

## 前提条件

| | |
|---|---|
| 机器人 | **宇树 R1 EDU**，板载 Jetson Orin NX 16 GB |
| 系统 | **JetPack 5.1.1**（L4T R35.3.1）：TensorRT 8.5.2、CUDA 11.4、Ubuntu 20.04、Python 3.8 |
| ROS | **ROS 2 foxy**，装在 `/opt/ros/foxy` |
| 宇树 SDK | **unitree_sdk2** 已编译并安装到 `/usr/local`（有 `/usr/local/include/unitree/` 和 `/usr/local/lib/libunitree_sdk2.a`）。硬件桥要链接它。本发布不包含它 |
| 磁盘 / 内存 | 约 5 GB 空闲磁盘；模型加载时约需 2.5 GB 空闲内存 |
| 网络 | 机器人端不需要；需要一台电脑下载发布包，再用 SSH 拷到机器人上 |
| 人和装备 | 急停遥控在手；头几次要吊装；最好有第二个人在场 |

## 安装

**1. 在电脑上：下载并校验**

```bash
sha256sum -c SHA256SUMS          # 两个压缩包都显示 OK
scp release_v1.1.0.tar.gz qwen3-1.7b_r1-robot-v1.1.0.tar.gz unitree@<机器人IP>:~/
```
保持 `.tar.gz` 格式拷贝。zip 会丢掉文件的可执行权限。

**2. 在机器人上：两个包解到同一处**

```bash
cd ~
tar xzf release_v1.1.0.tar.gz
tar xzf qwen3-1.7b_r1-robot-v1.1.0.tar.gz      # 会补上 r1-robot-v1.1.0/llm/models/
cd ~/r1-robot-v1.1.0
bash selftest.sh                                 # 每行都应是 ok；不动机器人，也不需要起栈
```

**3. 编译并安装策略（约 2 分钟）**

```bash
export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1
source ~/r1-robot-v1.1.0/deploy/env.sh
bash ~/r1-robot-v1.1.0/policy_pack/install_bundle.sh \
     ~/r1-robot-v1.1.0/bundles/2026-08-19_11-03-32_week04_nohead
```

`install_bundle.sh` 共八步，任何一步失败都会停下：
1. 校验策略包；
2. 预览会改什么；
3. 放置文件；
4. 生成关节映射和配置；
5. `colcon build` 编译硬件桥和策略节点（约 30 秒）；
6. **在这台机器人上**构建 TensorRT 引擎（约 25 秒）；
7. **拿引擎的输出对照包里的参考数据**（parity 闸门）。实验室那台机器人的 `max_abs` 是 1.144e-05，上限是 1e-3；
8. 写出 `deploy/artifacts/installed.json`。

最后几行会打印引擎路径和指纹。指纹只用来标识"是哪个文件"，每次构建都会不同，因为 TensorRT 会按实测耗时挑选内核。数值是否正确由 parity 证明。

**4. 启动语言模型并检查**

```bash
bash ~/r1-robot-v1.1.0/llm/ollama_ctl.sh start
bash ~/r1-robot-v1.1.0/llm/ollama_ctl.sh warm
# 应看到：health: OK -- qwen3:1.7b answered in X s, and the plan is right
bash ~/r1-robot-v1.1.0/llm/ollama_ctl.sh status   # 模型应显示 on GPU（约 1.5 GB）
```

- 服务只监听 `127.0.0.1:11434`，以当前用户身份运行，不需要 sudo，也不用 systemd。
- 停止用 `ollama_ctl.sh stop`，重启用 `restart`。
- 机器人运动时不要开 `ollama_ctl.sh watchdog`：它每 30 秒让模型生成一次回答，而生成会和行走策略争用 GPU。

## 使用

### 每个终端先执行

```bash
export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1      # 每个终端都要，而且要在 source env.sh 之前
cd ~/r1-robot-v1.1.0/deploy && source env.sh
M=~/r1-robot-v1.1.0/mission_ctl/r1_mission_cli.py
```
不同终端的 `ROS_DOMAIN_ID` 不一致时，它们互相看不见，表现为栈对指令毫无反应。

### 每次上电、机器人动之前

1. **用手柄把机器人切到开发者模式。** 否则宇树自带的运控会和本栈同时给电机下指令，表现很像策略不行：晃、抖、关节异响。
2. **锁频**（需要 sudo，重启后失效）：`bash $R1_DEPLOY_ROOT/tools/w08_preflight.sh --lock-clocks`，必须 0 FAIL。不锁频的话，测出来的时延不可比。
3. 头几次把机器人吊在吊装上，绳子留一点松量。

### 起栈（终端 1）

**先关输出。** 电机不接收任何指令，其余部分照常运行：

```bash
ros2 launch r1_hw_bridge r1_stack.launch.py \
    engine:=$R1_DEPLOY_ROOT/artifacts/policy_fp32.plan iface:=eth10 \
    enable_output:=false kp_scale:=1.3 kd_scale:=1.0
```
约 5 秒后应看到 `RUNNING | obs 50.0 Hz ... cmd 500.0 Hz`，且没有 `DEGRADED`。确认后按 Ctrl-C，再用 `enable_output:=true` 重新起栈。

- **数值参数要带小数点**（`1.3`、`1.0`、`55.0`）。写成整数（如 `kp_scale:=1`）会让硬件桥在启动时直接退出。
- `iface` 是机器人接在 `192.168.123.x` 网段上的网卡，实验室那台是 `eth10`。用 `ip -br addr` 查自己的。
- `kp_scale` 是关节刚度的倍数。我们的真机在 1.10–1.50 范围内能稳定行走；训练值 1.0 恰好在这个范围的边缘，所以实验室演示用 **1.3**。

### 用英文下指令（终端 2）

```bash
python3 $M capability                      # 机器人能做什么（读自已安装的策略包）
python3 $M ask "Walk forward for 3 seconds at 0.2 meters per second."
python3 $M ask "Turn left 90 degrees."
python3 $M ask "Stand for 3 seconds, then walk forward 1 meter at 0.3 meters per second, then turn right 90 degrees."
```

`ask` 的流程：
1. GPU 上的模型把句子转写成步骤，约 1–4 秒；
2. 代码逐步对照策略的训练范围做检查；
3. 打印计划，包括速度、时间和角度；
4. **你按 `y` 或 `N`。按 `y` 之前先读一遍计划**；
5. 机器人执行；
6. 输出一段简短回报，例如："Walked forward for 3.0 s at 0.20 m/s: about 0.6 m"。

| 能做 | 不能做（会拒绝并说明原因） |
|---|---|
| 以 0–1.0 m/s 向前走 | 后退、横移 |
| 原地转到指定角度，最快 0.5 rad/s（约 29°/s），用 IMU 航向闭环 | 边走边转 |
| 原地站立 | 持续不到 2 秒的一步，或小到做不好的转向（例如 10°）；一个计划超过 20 步、120 秒或 30 米 |
| 走指定距离，但只是近似：时间 × 指令速度，没有里程计 | 测量实际走了多远 |

- **每句 `ask` 都是独立的。** 机器人不记得上一句，所以"again""back""the other way"这类说法会被拒绝。每句都要说清方向和数值。
- 每个距离、时间、角度都必须**在句子里说出来，并带单位**。模型自己补出来的数字会被丢弃，整个计划随之被拒。
- `--dry-run` 只仿真并打印指令序列，机器人不动。
- `python3 $M run "walk 3s@0.2; turn left 90"` 不经过模型，得到的是同样的计划。
- 每次 `ask` 是一个进程、一个计划。进程结束后，硬件桥的 500 ms 失联保护会把速度指令归零。

### 停下

| 方式 | 效果 |
|---|---|
| 在 `ask` 终端按 **Ctrl-C** | 立刻发零速度，并报告这一步走到了哪里；机器人原地保持平衡 |
| `ask` 进程被杀或崩溃 | 500 ms 收不到指令，硬件桥自动发零速度 |
| 硬件桥检测到故障（传感数据过期、策略不再回应、目标值非有限数、控制频率连续 3 秒低于 45 Hz） | 进入 **DEGRADED**，电机转为阻尼。停栈（终端 1 按 Ctrl-C）后重新起栈 |
| 任何意外情况 | **急停** |

## 常见问题

| 现象 | 原因与处理 |
|---|---|
| `selftest.sh` 报缺模型 | 第二个压缩包没在 `~/` 下解压（它应生成 `r1-robot-v1.1.0/llm/models/`） |
| `install_bundle.sh` 第 5 步报 `unitree/...: No such file` | `/usr/local` 下没有装 unitree_sdk2 |
| 第 7 步 `PARITY FAILED` | 不要运行这个引擎。检查系统是不是 JetPack 5.1.1 / TensorRT 8.5.2 |
| 硬件桥启动即退出：`InvalidParameterTypeException` | 某个数值启动参数写成了整数，改成 `1.3`、`55.0` 这样的写法 |
| 栈在跑，但指令没反应，或 `ask` 一直等硬件桥 | 各终端的 `ROS_DOMAIN_ID` / `ROS_LOCALHOST_ONLY` 不一致，或者没有 `source env.sh` |
| `ask` 拒绝：已有其他发布者在 `~/cmd_vel` 上 | 还有一个 `ask` / `run` 在运行。同一时间只能有一个在下指令 |
| `ollama_ctl.sh warm` 报模型没回答 | 执行 `bash llm/ollama_ctl.sh restart`。运行时偶尔会起来了却不服务（llama.cpp 在 Orin 上的已知问题），重启即可 |
| 行走时晃、关节异响或抖动 | 没切开发者模式，宇树自带运控也在给电机下指令 |
| `ros2 topic list` 崩溃（`bad_alloc`） | 这台机器上 ROS foxy 和宇树 SDK 的 DDS 库存在已知冲突。本栈除 `ros2 launch` 外不需要 `ros2` 命令行 |

策略推理和语言模型共用 GPU。模型生成回答期间，策略推理最长约 5 ms（控制周期是 20 ms），硬件桥的各项硬限仍然守得住。而且 `ask` 只在机器人**动之前**生成。

## 卸载

```bash
bash ~/r1-robot-v1.1.0/llm/ollama_ctl.sh stop
rm -rf -- ~/r1-robot-v1.1.0
```
这个目录之外没有安装任何东西。

## 许可证

代码采用 Apache-2.0（见 `LICENSE`、`NOTICE`）。Ollama 运行时、它附带的 CUDA 库和 Qwen3 模型各自保留原许可证，见 `THIRD_PARTY_NOTICES.md`。
