# R1 robot stack v1.1.0: English commands for a Unitree R1, on the robot

This archive is everything that runs **on the robot** (a Unitree R1 EDU, onboard Jetson
Orin NX). With it you can:

- build and install the walking policy, then check it against its reference numbers;
- start the control stack;
- type an English instruction such as
  `python3 $M ask "Walk forward for 3 seconds at 0.2 meters per second."`

A small language model running on the robot's own GPU turns that sentence into a plan.
Deterministic code checks the plan against what the policy was trained to do and shows it
to you. The robot moves only after you press `y`. No internet connection is needed on
the robot.

Source, design notes and results: <https://github.com/kedianwen/LDRQ> (this release is
built from its tag `v1.1.0`).

> **Read the safety section before anything moves.** This is research code for a robot
> that can fall and can hurt people. Run it first with the robot hanging in a gantry,
> with the emergency stop in your hand.

## Contents

| release asset | size | what it is |
|---|---|---|
| `release_v1.1.0.tar.gz` | ~0.5 GB | this README, the code, the policy bundle, and the Ollama language-model server for JetPack 5 |
| `qwen3-1.7b_r1-robot-v1.1.0.tar.gz` | ~1.3 GB | the language model, qwen3:1.7b (Q4_K_M), in Ollama's format |
| `SHA256SUMS` | — | checksums of the two archives |

Both archives extract into the same directory, `r1-robot-v1.1.0/`:

```
r1-robot-v1.1.0/
  README.md, README_zh.md   this file (English / 中文)
  selftest.sh               checks that need no robot motion
  deploy/                   C++/ROS 2 hardware bridge and TensorRT policy node (built on the robot)
  bundles/<run_id>/         the trained policy: ONNX, interface, gains, command envelope, parity fixture
  policy_pack/              installs a bundle: generate configs, build, build the engine, parity gate
  mission_ctl/              the command layer: run / ask / capability
  llm/                      ollama_ctl.sh, the Ollama runtime (llm/ollama/), the model (llm/models/)
  LICENSE, NOTICE, THIRD_PARTY_NOTICES.md
```

The policy is `2026-08-19_11-03-32_week04_nohead`: 425 proprioceptive inputs → 24 joint
targets at 50 Hz, trained in Isaac Lab.

## Requirements

| | |
|---|---|
| robot | **Unitree R1 EDU** with the onboard Jetson Orin NX 16 GB |
| system | **JetPack 5.1.1** (L4T R35.3.1): TensorRT 8.5.2, CUDA 11.4, Ubuntu 20.04, Python 3.8 |
| ROS | **ROS 2 foxy** in `/opt/ros/foxy` |
| Unitree SDK | **unitree_sdk2** built and installed under `/usr/local` (`/usr/local/include/unitree/`, `/usr/local/lib/libunitree_sdk2.a`). The bridge is compiled against it. It is not part of this release |
| disk / memory | about 5 GB free disk; about 2.5 GB free memory while the model is loaded |
| network | none on the robot; a PC to download the release and copy it over SSH |
| people and gear | an emergency stop in hand; a gantry for the first runs; a second person is better |

## Install

**1. On a PC: download and check.**

```bash
sha256sum -c SHA256SUMS          # both archives: OK
scp release_v1.1.0.tar.gz qwen3-1.7b_r1-robot-v1.1.0.tar.gz unitree@<robot-ip>:~/
```
Copy them as `.tar.gz`. A zip loses the files' execute permissions.

**2. On the robot: extract both archives into the same place.**

```bash
cd ~
tar xzf release_v1.1.0.tar.gz
tar xzf qwen3-1.7b_r1-robot-v1.1.0.tar.gz      # adds r1-robot-v1.1.0/llm/models/
cd ~/r1-robot-v1.1.0
bash selftest.sh                                 # every line "ok"; moves nothing, needs no stack
```

**3. Build and install the policy (about 2 minutes).**

```bash
export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1
source ~/r1-robot-v1.1.0/deploy/env.sh
bash ~/r1-robot-v1.1.0/policy_pack/install_bundle.sh \
     ~/r1-robot-v1.1.0/bundles/2026-08-19_11-03-32_week04_nohead
```
`install_bundle.sh` runs eight checked steps. Any failure stops it.
1. Validate the bundle.
2. Show what will change.
3. Stage the files.
4. Generate the joint map and the configs.
5. `colcon build` the bridge and the policy node (about 30 s).
6. Build the TensorRT engine **on this robot** (about 25 s).
7. **Check the engine's outputs against the bundle's reference** (the parity gate; on the
   lab robot `max_abs` was 1.144e-05 against a 1e-3 limit).
8. Write `deploy/artifacts/installed.json`.

The last lines print the engine path and its fingerprint. The fingerprint identifies the
file. It differs from build to build, because TensorRT picks kernels by timing them.
Parity is what proves the numbers are right.

**4. Start the language model and check it.**

```bash
bash ~/r1-robot-v1.1.0/llm/ollama_ctl.sh start
bash ~/r1-robot-v1.1.0/llm/ollama_ctl.sh warm
# expect: "health: OK -- qwen3:1.7b answered in X s, and the plan is right"
bash ~/r1-robot-v1.1.0/llm/ollama_ctl.sh status   # the model should show "on GPU" (~1.5 GB)
```
The server listens on `127.0.0.1:11434` only and runs as your user: no sudo, no
systemd. `ollama_ctl.sh stop` stops it, and `restart` restarts it. Leave
`ollama_ctl.sh watchdog` off while the robot moves: it generates an answer every 30 s,
and generation shares the GPU with the walking policy.

## Use

### In every terminal, first

```bash
export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1      # before env.sh, in EVERY terminal
cd ~/r1-robot-v1.1.0/deploy && source env.sh
M=~/r1-robot-v1.1.0/mission_ctl/r1_mission_cli.py
```
Terminals with different `ROS_DOMAIN_ID` settings cannot see each other. The symptom is
a stack that seems to ignore every command.

### Before the robot moves, every power-on

1. **Put the robot in developer mode** with its handheld remote. Otherwise Unitree's own
   motion controller keeps commanding the motors at the same time as this stack. The
   result looks like a bad policy: sway, tremor, grinding joints.
2. **Lock the clocks** (needs sudo, lasts until reboot): `bash $R1_DEPLOY_ROOT/tools/w08_preflight.sh --lock-clocks`.
   It must report 0 FAIL. Without it, timing numbers are not comparable.
3. Hang the robot in the gantry, with a little slack in the rope, for the first runs.

### Start the stack (terminal 1)

**First with output off.** The motors receive nothing; everything else runs:

```bash
ros2 launch r1_hw_bridge r1_stack.launch.py \
    engine:=$R1_DEPLOY_ROOT/artifacts/policy_fp32.plan iface:=eth10 \
    enable_output:=false kp_scale:=1.3 kd_scale:=1.0
```
Wait for `RUNNING | obs 50.0 Hz ... cmd 500.0 Hz`, about 5 s, with no `DEGRADED`. Then
press Ctrl-C and start it again with `enable_output:=true`.

- **Write numeric arguments with a decimal point** (`1.3`, `1.0`, `55.0`). An integer
  such as `kp_scale:=1` stops the bridge at start-up.
- `iface` is the robot's network interface on the `192.168.123.x` network: `eth10` on
  the lab robot. Check yours with `ip -br addr`.
- `kp_scale` multiplies the joint stiffness. Our real robot walked stably from 1.10 to
  1.50. The trained value, 1.0, sits on the edge of that range, so the lab demos used
  **1.3**.

### Command it in English (terminal 2)

```bash
python3 $M capability                      # what the robot can do, from the installed bundle
python3 $M ask "Walk forward for 3 seconds at 0.2 meters per second."
python3 $M ask "Turn left 90 degrees."
python3 $M ask "Stand for 3 seconds, then walk forward 1 meter at 0.3 meters per second, then turn right 90 degrees."
```

What `ask` does, in order:
1. The model, on the GPU, transcribes the sentence into steps. This takes about 1–4 s.
2. Code checks every step against the policy's training range.
3. The plan is printed with its speeds, times and angles.
4. **You press `y` or `N`.** Read the plan before you press `y`.
5. The robot executes the plan.
6. A short report says what was done, for example: "Walked forward for 3.0 s at 0.20 m/s:
   about 0.6 m (time x commanded speed; ...)".

| it can | it cannot (it refuses and says why) |
|---|---|
| walk forward at 0 to 1.0 m/s | walk backward; step sideways |
| turn on the spot, up to 0.5 rad/s (about 29°/s), to a given angle; the turn is closed on the IMU heading | turn while walking |
| stand still | a step that would last under 2 s, or a turn too small to do well (10°, for example); more than 20 steps, 120 s or 30 m in one plan |
| walk a distance, approximately: time × commanded speed, with no odometry | measure how far it actually went |

- **Every `ask` stands alone.** The robot does not remember the previous sentence, so
  "again", "back" or "the other way" are refused. Always state the direction and the
  number.
- Every distance, time and angle must be **said in the sentence, with its unit**. A
  number the model makes up is dropped, and the plan is refused.
- `--dry-run` simulates and prints the command trace without moving anything.
  `python3 $M run "walk 3s@0.2; turn left 90"` gives the same plans without the model.
- Each `ask` is one process and one plan. When it ends, the bridge's 500 ms deadman takes
  the velocity command to zero.

### Stopping

| what | effect |
|---|---|
| **Ctrl-C** in the `ask` terminal | sends zero velocity at once and reports how far the step got; the robot keeps balancing in place |
| `ask` process killed or crashed | no command for 500 ms → the bridge commands zero velocity |
| the bridge detects a fault (stale sensor data, the policy stops answering, a non-finite target, the control rate under 45 Hz for 3 s) | **DEGRADED**: motors go to damping. Stop the stack (Ctrl-C in terminal 1) and start it again |
| anything unexpected | **the emergency stop** |

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `selftest.sh` reports a missing model | the second archive was not extracted into `~/` (it must create `r1-robot-v1.1.0/llm/models/`) |
| `install_bundle.sh` fails at step 5 with `unitree/...: No such file` | unitree_sdk2 is not installed under `/usr/local` |
| `PARITY FAILED` at step 7 | do not run the engine. Check that this is JetPack 5.1.1 / TensorRT 8.5.2 |
| bridge dies at start: `InvalidParameterTypeException` | a numeric launch argument was written as an integer; use `1.3`, `55.0` |
| stack runs, but commands have no effect, or `ask` waits for the bridge | `ROS_DOMAIN_ID` / `ROS_LOCALHOST_ONLY` differ between terminals, or `env.sh` was not sourced |
| `ask` refuses: "another publisher on ~/cmd_vel" | another `ask` / `run` is still running. Only one may command the robot at a time |
| `ollama_ctl.sh warm` says the model did not answer | `bash llm/ollama_ctl.sh restart`. The runtime occasionally starts without serving (a known llama.cpp issue on Orin); a restart fixes it |
| walking sways, joints grind or tremble | developer mode is off: Unitree's controller is also commanding the motors |
| `ros2 topic list` crashes (`bad_alloc`) | a known clash between ROS foxy's and the Unitree SDK's DDS libraries on this robot. Nothing in this stack needs the `ros2` CLI except `ros2 launch` |

Policy inference shares the GPU with the language model. While the model is generating
an answer, inference takes up to about 5 ms of the 20 ms control step. The bridge's
limits still hold, and `ask` only generates **before** the robot moves.

## Remove

```bash
bash ~/r1-robot-v1.1.0/llm/ollama_ctl.sh stop
rm -rf -- ~/r1-robot-v1.1.0
```
Nothing was installed outside this directory.

## Licences

The code is Apache-2.0 (`LICENSE`, `NOTICE`). The Ollama runtime, the CUDA libraries it
bundles and the Qwen3 model keep their own licences; see `THIRD_PARTY_NOTICES.md`.
