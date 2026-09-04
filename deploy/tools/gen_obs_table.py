#!/usr/bin/env python3
"""Build the W06 observation-assembly consistency table -- and gate on it.

The W06 plan asks for a table cross-checking the deployment observation against
what training produced. A table typed by hand is worth very little: it records
what someone believed on the day they wrote it, and it goes stale the moment a
constant moves. So this script *derives* the table from the four files that
actually decide the behaviour, and returns non-zero if any two of them disagree:

  1. deploy/interface/policy_interface.json          <- training, exported
  2. .../r1_hw_bridge/include/r1_hw_bridge/joint_map.hpp  <- what the bridge reads
  3. .../r1_policy_runner/config/policy_interface.yaml    <- what the policy node loads
  4. .../r1_hw_bridge/config/bridge.yaml                  <- rate the bridge runs at

Every one of these has an independent path onto the robot (a re-export, a
regenerated header, a hand-edited yaml, a launch override), which is exactly why
they can drift apart without anything failing loudly.

  python3 gen_obs_table.py                    # print the table
  python3 gen_obs_table.py -o ../interface/obs_consistency_table.md
  python3 gen_obs_table.py --check            # gate only, no output
"""
import argparse
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
DEPLOY = HERE.parent
JSON_PATH = DEPLOY / "interface" / "policy_interface.json"
HPP_PATH = DEPLOY / "ros2_ws/src/r1_hw_bridge/include/r1_hw_bridge/joint_map.hpp"
PYAML_PATH = DEPLOY / "ros2_ws/src/r1_policy_runner/config/policy_interface.yaml"
BYAML_PATH = DEPLOY / "ros2_ws/src/r1_hw_bridge/config/bridge.yaml"

# The slot set observed live on the robot by `probe_lowstate --seconds`
# (2026-09-02): the slots that carry a real motor. Independent of every file
# above -- it came off the hardware -- so comparing the mapped set against it is
# a free 26-way check that no per-joint wiggle test can give you.
MEASURED_LIVE_SLOTS = set(range(14)) | {15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 29, 30}

# Joints with a direct wiggle measurement behind them (see
# experiment_record/Week06_关节映射与IMU约定_实测定稿.md). Everything else is
# covered transitively by the vendor enum + the set check above.
MEASURED_ANCHORS = {
    "left_hip_pitch_joint": 0,
    "left_knee_joint": 3,
    "left_ankle_roll_joint": 5,
    "right_hip_pitch_joint": 6,
    "left_shoulder_roll_joint": 16,
    "head_pitch_joint": 29,
}

TOL = 1e-6
problems = []


def fail(msg):
    problems.append(msg)


def close(a, b, tol=TOL):
    return abs(float(a) - float(b)) <= tol


def parse_cpp_array(text, name):
    """Pull `... name[...] = { 1, 2, 3 };` out of the generated header."""
    m = re.search(re.escape(name) + r"\s*(?:\[[^\]]*\])?\s*=\s*\{(.*?)\}\s*;", text, re.S)
    if not m:
        raise SystemExit(f"[fail] {name} not found in {HPP_PATH}")
    body = re.sub(r"//[^\n]*", "", m.group(1))
    return [t.strip().rstrip("fF") for t in body.split(",") if t.strip()]


def parse_cpp_scalar(text, name):
    m = re.search(re.escape(name) + r"\s*=\s*([0-9]+)", text)
    if not m:
        raise SystemExit(f"[fail] {name} not found in {HPP_PATH}")
    return int(m.group(1))


def parse_yaml_seq(text, key):
    """These yamls are generated and flat; a real yaml parser is not worth the
    dependency on a robot with no network."""
    m = re.search(rf"^\s*{re.escape(key)}\s*:\s*\[(.*?)\]\s*$", text, re.M | re.S)
    if not m:
        return None
    return [t.strip().strip('"') for t in m.group(1).split(",") if t.strip()]


def parse_yaml_scalar(text, key):
    m = re.search(rf"^\s*{re.escape(key)}\s*:\s*([^\s#]+)", text, re.M)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", help="write markdown here instead of stdout")
    ap.add_argument("--check", action="store_true", help="only run the checks")
    args = ap.parse_args()

    j = json.loads(JSON_PATH.read_text())
    hpp = HPP_PATH.read_text()
    pyaml = PYAML_PATH.read_text()
    byaml = BYAML_PATH.read_text()

    art_names = j["articulation"]["joint_names"]
    art_def = j["articulation"]["default_joint_pos"]
    act_names = j["action"]["joint_names"]
    act_ids = j["action"]["joint_ids_in_articulation"]
    act_scale = j["action"]["scale"]
    act_def = j["action"]["default_pos"]
    terms = [(t["name"], t["width"]) for t in j["observation"]["terms"]]
    hist = j["observation"]["history_length"]
    rate = j["control"]["control_rate_hz"]

    n_joints = parse_cpp_scalar(hpp, "kNumJoints")
    n_slots = parse_cpp_scalar(hpp, "kNumSlots")
    n_act = parse_cpp_scalar(hpp, "kNumActions")
    slots = [int(x) for x in parse_cpp_array(hpp, "kJointSlot")]
    hdefs = [float(x) for x in parse_cpp_array(hpp, "kDefaultPos")]
    a2a = [int(x) for x in parse_cpp_array(hpp, "kActionToArt")]

    # ---- C1  control rate: three files, one number ----------------------
    p_rate = float(parse_yaml_scalar(pyaml, "control_rate_hz"))
    b_rate = float(parse_yaml_scalar(byaml, "control_rate_hz"))
    if not (close(rate, p_rate) and close(rate, b_rate)):
        fail(f"C1 control_rate_hz: json={rate} policy.yaml={p_rate} bridge.yaml={b_rate}")

    # ---- C2  history length --------------------------------------------
    p_hist = int(parse_yaml_scalar(pyaml, "history_length"))
    if p_hist != hist:
        fail(f"C2 history_length: json={hist} policy.yaml={p_hist}")

    # ---- C3  term names and widths, in order ----------------------------
    y_names = parse_yaml_seq(pyaml, "obs_term_names")
    y_widths = [int(x) for x in parse_yaml_seq(pyaml, "obs_term_widths")]
    if y_names != [n for n, _ in terms]:
        fail(f"C3 obs_term_names: json={[n for n,_ in terms]} yaml={y_names}")
    if y_widths != [w for _, w in terms]:
        fail(f"C3 obs_term_widths: json={[w for _,w in terms]} yaml={y_widths}")

    # ---- C4  the dimensions the engine will reject if wrong -------------
    frame = sum(w for _, w in terms)
    if frame != j["observation"]["frame_dim"]:
        fail(f"C4 frame_dim: sum(widths)={frame} json={j['observation']['frame_dim']}")
    if frame * hist != j["observation"]["total_dim"]:
        fail(f"C4 total_dim: {frame}x{hist}={frame*hist} json={j['observation']['total_dim']}")

    # ---- C5  26 everywhere ---------------------------------------------
    if not (len(art_names) == n_joints == dict(terms)["joint_pos"] == dict(terms)["joint_vel"]):
        fail(f"C5 joint count: art={len(art_names)} hpp={n_joints} terms={dict(terms)['joint_pos']}/{dict(terms)['joint_vel']}")

    # ---- C6  the defaults the bridge subtracts --------------------------
    for i, (a, b) in enumerate(zip(art_def, hdefs)):
        if not close(a, b, 1e-5):
            fail(f"C6 kDefaultPos[{i}] ({art_names[i]}): json={a} hpp={b}")

    # ---- C7  24 everywhere ---------------------------------------------
    y_scale = [float(x) for x in parse_yaml_seq(pyaml, "action_scale")]
    y_def = [float(x) for x in parse_yaml_seq(pyaml, "default_joint_pos")]
    if not (j["action"]["dim"] == n_act == dict(terms)["actions"] == len(y_scale) == len(y_def)):
        fail(f"C7 action dim: json={j['action']['dim']} hpp={n_act} term={dict(terms)['actions']} yaml={len(y_scale)}/{len(y_def)}")

    # ---- C8  action index -> articulation index --------------------------
    # The trap this exists for: action[i] and joint_pos[i] are DIFFERENT joints.
    if a2a != act_ids:
        fail(f"C8 kActionToArt != json joint_ids_in_articulation")
    for a, art in enumerate(a2a):
        if art_names[art] != act_names[a]:
            fail(f"C8 action[{a}] is {act_names[a]} but kActionToArt says art {art} = {art_names[art]}")

    # ---- C9/C10  scale and default reach the node unchanged -------------
    for a in range(len(y_scale)):
        if not close(y_scale[a], act_scale[a], 1e-5):
            fail(f"C10 action_scale[{a}] ({act_names[a]}): json={act_scale[a]} yaml={y_scale[a]}")
        if not close(y_def[a], act_def[a], 1e-5):
            fail(f"C9 default_joint_pos[{a}] ({act_names[a]}): json={act_def[a]} yaml={y_def[a]}")
        if not close(y_def[a], hdefs[a2a[a]], 1e-5):
            fail(f"C9 default mismatch across vectors for {act_names[a]}: policy={y_def[a]} bridge={hdefs[a2a[a]]}")

    # ---- C11  slots: unique, in range, and equal to what the robot showed
    if len(set(slots)) != len(slots):
        dup = [s for s in set(slots) if slots.count(s) > 1]
        fail(f"C11 duplicate slots: {dup}")
    if any(s < 0 or s >= n_slots for s in slots):
        fail(f"C11 slot out of range 0..{n_slots-1}")
    if set(slots) != MEASURED_LIVE_SLOTS:
        fail(f"C11 mapped slots != measured LIVE set; "
             f"extra={sorted(set(slots)-MEASURED_LIVE_SLOTS)} "
             f"missing={sorted(MEASURED_LIVE_SLOTS-set(slots))}")
    for name, slot in MEASURED_ANCHORS.items():
        if name not in art_names:
            fail(f"C11 anchor joint {name} not in articulation")
        elif slots[art_names.index(name)] != slot:
            fail(f"C11 anchor {name}: measured slot {slot}, table says {slots[art_names.index(name)]}")

    # ---- C12  flatten order --------------------------------------------
    if "per-term" not in j["observation"]["flatten"]:
        fail(f"C12 flatten is '{j['observation']['flatten']}', not per-term")
    if "empirical_normalization=false" not in j["observation"]["normalizer"]:
        fail(f"C12 normalizer is '{j['observation']['normalizer']}', not raw")

    if problems:
        print(f"[FAIL] {len(problems)} inconsistency(ies):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    if args.check:
        print("[ok] 12/12 consistency checks pass")
        return 0

    md = render(j, terms, hist, frame, rate, slots, art_names, art_def, a2a, act_names)
    if args.out:
        pathlib.Path(args.out).write_text(md)
        print(f"[ok] 12/12 checks pass -> {args.out}")
    else:
        sys.stdout.write(md)
    return 0


def render(j, terms, hist, frame, rate, slots, art_names, art_def, a2a, act_names):
    L = []
    w = L.append
    w("# W06 · 观测装配一致性对照表\n")
    w("> **本文件由 `deploy/tools/gen_obs_table.py` 生成，不要手改。**")
    w("> 生成器同时是**门禁**：四个文件里任意两个不一致就非零退出。")
    w("> 手抄的对照表只记录写它那天的信念；这个每次运行都重新核对。\n")
    w(f"训练来源：`{j['run']}` / `{j['task']}`\n")
    w("交叉核对的四个文件（各自有独立的上机路径，所以能各自漂移）：\n")
    w("| # | 文件 | 它决定什么 |")
    w("|---|---|---|")
    w("| 1 | `deploy/interface/policy_interface.json` | 训练端真值，由 `dump_interface.py` 导出 |")
    w("| 2 | `r1_hw_bridge/include/r1_hw_bridge/joint_map.hpp` | bridge 从 35 槽里取哪 26 个、减哪些默认角 |")
    w("| 3 | `r1_policy_runner/config/policy_interface.yaml` | 策略节点的历史长度、动作缩放、默认角 |")
    w("| 4 | `r1_hw_bridge/config/bridge.yaml` | bridge 实际跑的频率 |")
    w("")
    w("---\n")
    w("## 1 · 逐项对照\n")
    w("| 项 | 训练端 | 部署端实现 | 谁保证一致 |")
    w("|---|---|---|---|")
    w(f"| 控制周期 | {j['control']['sim_dt']} × {j['control']['decimation']} = "
      f"{j['control']['control_dt']*1000:.0f} ms（{rate:g} Hz） | bridge `control_rate_hz`；"
      f"命令线程独立跑 500 Hz | C1 三文件同值 + 节点自报速率 |")
    w(f"| 历史长度 | {hist} 帧 | `ObsAssembler`（在**策略节点**内，不在 bridge） | C2 |")
    w(f"| 展平顺序 | {j['observation']['flatten']} | `ObsAssembler::Emit()` 外层遍历项、内层遍历帧 | C12 + `test_obs_assembler` |")
    w(f"| 帧长 / 总长 | {frame} / {frame*hist} | 引擎输入维度不符则**拒绝启动** | C4 + 运行期 |")
    w(f"| 归一化 | {j['observation']['normalizer']} | 不做任何缩放，原值入网 | C12 |")
    w("| `base_ang_vel` (3) | 机体系角速度，rad/s | `LowState.imu_state.gyroscope`，直接透传 | — |")
    w("| `projected_gravity` (3) | `quat_rotate_inverse(q, (0,0,-1))` | `QuatRotateInverse()`，q 按 **(w,x,y,z)** 取 | 实测：直立时 ≈ (0,0,−1)，加速度计独立佐证 0.04 m/s² |")
    w("| `velocity_commands` (3) | 采样出的指令 | `~/cmd_vel`；**超时 500 ms 归零** | 遥控断了不能让机器人接着走 |")
    w("| `joint_pos` (26) | **相对默认姿态**，articulation 顺序 | `motor[kJointSlot[j]].q - kDefaultPos[j]` | C5/C6 + 阶段 3.4 |")
    w("| `joint_vel` (26) | **绝对值**，同顺序 | `motor[kJointSlot[j]].dq`，不减任何东西 | C5 |")
    w("| `actions` (24) | 上一周期策略原始输出（**缩放前**） | bridge 订阅 `~/action` 回填，不是 `joint_target` | C7/C8 |")
    w(f"| 动作映射 | `{j['action']['target_formula']}` | 在**策略节点**内做，bridge 收到的已是目标角 | C9/C10 |")
    w("")
    w("### 三个反复踩的坑\n")
    w("1. **展平是按项不是按帧。** 两种排法都是 425 个浮点数，错的那种照样加载、照样跑。")
    w("2. **`actions` 项喂的是缩放前的原始输出。** 训练时记录的就是网络输出本身；")
    w("   回填 `joint_target` 数值量级完全不同，且不报错。")
    # Pick a real example rather than a fixed index: at some indices the two
    # vectors happen to agree, which would illustrate the opposite point.
    ex = next(i for i in range(len(act_names)) if act_names[i] != art_names[i])
    w(f"3. **`action[i]` 和 `joint_pos[i]` 不是同一个关节。** 例：`action[{ex}]` = "
      f"`{act_names[ex]}`（落在 articulation 第 {a2a[ex]} 位），"
      f"而 `joint_pos[{ex}]` = `{art_names[ex]}`。索引一样，关节不同。")
    w("")
    w("---\n")
    w("## 2 · 26 关节：槽位 / 默认角 / 动作索引\n")
    w("| art | 关节 | hg 槽 | 默认角 | action | 证据 |")
    w("|---:|---|---:|---:|---:|---|")
    rev = {art: a for a, art in enumerate(a2a)}
    for i, name in enumerate(art_names):
        a = rev.get(i)
        ev = "**实测**" if name in MEASURED_ANCHORS else "enum + 槽集"
        w(f"| {i} | `{name}` | {slots[i]} | {art_def[i]:+.3f} | "
          f"{a if a is not None else '—'} | {ev} |")
    w("")
    w(f"槽集 = 普查实测 LIVE 集，{len(MEASURED_ANCHORS)}/{len(art_names)} 条有直接掰动证据，"
      f"其余由厂商 enum 加这条全局集合等式覆盖。\n")
    w("---\n")
    w("## 3 · 这张表**不**覆盖什么\n")
    w("门禁比对的是**静态常量**。以下三件事只有在机器人上跑起来才能验：\n")
    w("- bridge 有没有照着 `kJointSlot` 取数（表自洽 ≠ 代码照做）→ 上机清单阶段 3.3")
    w("- 默认角有没有真的减掉 → 阶段 3.4")
    w("- 四元数分量顺序 → 已实测定稿，见 W06 实验记录第 4 节\n")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
