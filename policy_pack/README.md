# policy_pack · 版本 A：换策略仍能一键部署

把"换一个策略"从**改三个配置 + 改一个 C++ 头文件 + 重编**，变成**一条命令**。

```bash
# 开发机：导出之后，打包
python3 policy_pack/make_bundle.py

# 机器人：装包（八步，任一步失败即停，不留半装状态）
bash policy_pack/install_bundle.sh bundles/<run_id>
bash policy_pack/install_bundle.sh bundles/<run_id> --dry-run   # 只看会改什么
```

> **状态（2026-09-27）**：开发机上已测（12/12，配置生成与仓库逐字节一致）；
> **尚未上机**——`install_bundle.sh` 的第 5–8 步（colcon、建引擎、parity）只在机器人上才能跑，
> 是阶段 D 的验收内容。机器人是 Python 3.8，这里的脚本只在 3.10 上实跑过。

## 一个 bundle 是什么

```
bundles/<run_id>/
  policy.onnx               导出的策略（不是 .plan —— 见下）
  parity_fixture.bin        这个策略自己的数值锚点
  policy_interface.json     观测/动作契约：项名·宽度·history、动作映射·scale·default_pos
  actuator_gains.json       每关节 kp/kd/力矩上限（六组，不是一个标量）
  command_envelope.json     训练覆盖到的指令包线（从训练配置解析，不是从 bridge.yaml 抄）
  provenance.json           run_id、git sha、onnx sha256、生成日期与主机
  MANIFEST.sha256           上面每个文件的哈希
```

**bundle 里没有引擎。** TensorRT 的 `.plan` 编码了 TensorRT 版本、GPU 架构和在那台机器上
实测挑出来的 kernel tactic，不可移植。所以发过去的是 ONNX，引擎在 Orin 上建，
建完立刻用**这个包自带的 fixture** 做 parity —— 新策略在动关节之前先被证明"算得对"。

**bundle 里也没有电机槽位映射。** `kJointSlot` 是**机器人**的属性、是实测出来的
（厂家枚举里 `RightShoulderPitch = 19` 写成了 29，当槽位用会打到 head_pitch），
放进按策略走的文件等于邀请别人在 yaml 里"修正"它。它留在 `joint_map_measured.tsv`。

## 为什么是"重新生成 + 重编"，而不是"桥读参数"

`joint_map.hpp` 里的数组是 `constexpr`，所以**编译器**会拿它们的长度去核对
`kNumJoints` / `kNumActions`——长度不对是一个编译错误，而不是一台已经站着的机器人上的
运行期意外。实测这台 Orin 上两个包重编 **29 s**（`r1_hw_bridge` 24.3 s + `r1_policy_runner` 25.3 s，
并行）。这比去改 W06 以来唯一一直稳定的那个节点便宜。

## 八步做什么，以及每步在防什么

| 步 | 动作 | 防的是 |
|---|---|---|
| 1 | `verify_bundle.py` | 见下面的拒绝清单。**不写任何文件** |
| 2 | `gen_configs.py --check` | 让操作者在动手之前看到"会改什么"（注释变动会被明确标成注释变动） |
| 3 | 把 json/onnx/fixture 拷进 deploy 树 | —— |
| 4 | `gen_joints.py` + `gen_joint_map.py --header=` | 第二道独立校验：增益顺序 ≠ 关节顺序会在这里 `SystemExit` |
| 5 | `gen_configs.py` 写两个 yaml | 换了策略却留着上一个策略的包线 |
| 6 | `colcon build` | 生成的头文件编不过 = 长度/类型不对 |
| 7 | `r1_build_engine`（ONNX 未变则复用，`--force-engine` 强制重建） | —— |
| 8 | `r1_parity_check --tol 1e-3` + 写 `installed.json` | 引擎在这台机器上算错数 |

`installed.json` 记录 run_id、onnx sha256、bundle manifest 哈希、plan 体积与 fingerprint，
所以"现在机器人上跑的到底是哪个策略"是一个可以读出来的事实，不是一个需要回忆的事情。

## verify_bundle.py 会拒绝什么

`tests/run_tests.sh` 把下面每一条都真的造出来跑一遍（12/12 通过），
并且**断言拒绝的理由**，不只断言退出码——理由错的校验器会把人带向错误的方向。

| 坏 bundle | 为什么危险 |
|---|---|
| `history_length` 与 `total_dim` 不一致 | 不拦的话要等到 25 s 引擎建完、加载时才发现 |
| 观测项是桥**不会算**的（如 `base_lin_vel`） | 桥只会算六项；多出来的项会被 50 Hz 补零，读起来像 sim2real gap，能查一周 |
| 六项都对但**顺序**换了 | 维度全都加得上，纯静默错 |
| `actuator_gains.json` 少一个关节 / 按字母重排 | 桥按关节序号取增益，顺序不同 = 把踝的 kp 用在髋上 |
| 某条腿 `damping = 0` | 这就是 W06 那个"腿无阻尼"缺陷的文件形态 |
| 动作索引名字写 A、指向 B | 膝的指令发到肩，静默 |
| 包线 `lo > hi` | —— |
| 改了 json 但没重算 manifest | **最可能真发生的事故** |
| 包里多了一个没登记的文件 | 手改的 yaml 就是这样被装上去的 |
| 换了 onnx 但 provenance 没改 | 引擎与"记录在案的策略"不是一个东西 |

## 边界（这个版本做不到什么）

只有**桥已经会算的六个观测项**能靠配置部署。一个新策略如果加了 base 线速度、足底接触、
高度扫描，**必须改 C++**——正确行为是启动时报出那个项名并拒绝，而不是猜。
`verify_bundle.py` 里 `BRIDGE_TERMS` 那张表就是这条边界，改桥的组帧代码时要同步改它。
