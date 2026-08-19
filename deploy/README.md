# deploy/ — TensorRT + ROS 2 runner for the R1 policy

Closes Week04's deferred C++/ROS2 track and is the foundation W05 (ONNX export
+ TensorRT engine) and W06 (C++ ROS2 control node) build on. Self-contained:
nothing here needs sudo, and the training environment (`conda env_isaaclab`) is
never touched.

The policy it runs is `2026-08-19_11-03-32_week04_nohead` — 425-dim
observation, 24-dim action, 50 Hz.

## Quick start

```bash
./deploy/setup.sh                      # provision (idempotent; --check to verify only)
source deploy/env.sh                   # sets TENSORRT_ROOT/CUDART_ROOT, sources ROS 2
deploy/.venv/bin/python deploy/tools/check_env.py

cd deploy/ros2_ws && colcon build --packages-select r1_policy_runner && cd ../..
source deploy/env.sh                   # again, to overlay the built workspace

BIN=deploy/ros2_ws/install/r1_policy_runner/lib/r1_policy_runner
$BIN/r1_build_engine  --onnx deploy/artifacts/policy.onnx --plan deploy/artifacts/policy_fp32.plan
$BIN/r1_parity_check  --plan deploy/artifacts/policy_fp32.plan \
                      --fixture deploy/artifacts/parity_fixture.bin

ros2 launch r1_policy_runner policy_node.launch.py \
    engine:=$PWD/deploy/artifacts/policy_fp32.plan mode:=selftest
```

## What is here

| path | role |
|---|---|
| `setup.sh` | provisions TensorRT, its C++ headers and the CUDA runtime into `deploy/` |
| `env.sh` | source-able; picks dev-box vs JetPack paths, steps out of conda, sources ROS 2 |
| `tools/check_env.py` | version matrix; run it on both hosts and compare |
| `tools/dump_interface.py` | emits the policy↔robot contract from a live Isaac Lab env |
| `tools/make_fixture.py` | records PyTorch reference I/O for the parity check |
| `ros2_ws/src/r1_policy_runner/` | the ROS 2 package: engine build, parity check, node |
| `interface/policy_interface.{json,md}` | generated deployment contract (tracked) |
| `artifacts/` | ONNX, fixture, engines (git-ignored) |

Three executables:

- **`r1_build_engine`** — ONNX → serialised engine. Must run on the machine that
  will execute it.
- **`r1_parity_check`** — replays the fixture through the engine, compares
  against PyTorch, and profiles latency. **This is the gate**, not a nicety.
- **`r1_policy_node`** — the ROS 2 node. `mode:=subscribe` reads one 85-float
  sensor frame per cycle from `~/obs`; `mode:=selftest` self-drives at 50 Hz
  with no robot attached (bench rehearsal for W06's hanging dry-run).

## Two findings that change how W05/W06 must be done

### 1. TensorRT 10.3.0 silently computes the wrong answer on this host

The natural choice was TensorRT **10.3.0**, the version JetPack 6.1/6.2 ships
for Jetson Orin. On this box (Turing sm_75, driver 580.173) it builds an engine
that loads, runs at full speed, reports no error, and **returns wrong numbers** —
`max_abs 3.0e+01` against PyTorch, i.e. unrelated output.

It is not a precision effect and not a bad tactic. Reproduced identically from
the ONNX parser and from a network hand-built through the TensorRT API, at every
builder optimisation level 0–5, in FP32 and FP16, from both C++ and Python,
while **onnxruntime and PyTorch agree with each other to 1e-5**. Bisecting the
graph: any subgraph containing two ELU layers is correct, three is wrong.

TensorRT **10.7.0** matches PyTorch to `1.05e-05`. `setup.sh` pins it.

The consequence is the important part. The dev box and the robot are now on
**different TensorRT versions**, and the failure mode is invisible without a
numerical reference — no exception, no warning, full speed, plausible-looking
output. So:

> **`r1_parity_check` must be run again on the robot, on the engine built
> there, before the policy is ever allowed to drive a joint.** A green parity
> check on the dev box says nothing about the Jetson.

This is the same failure class as Week04's actuator-limit and head-tilt bugs:
a defect that no aggregate metric surfaces, found only by comparison against an
outside reference.

### 2. The microbenchmark overstates inference speed by ~8x

| measurement | p50 | p95 | p99 | max |
|---|---|---|---|---|
| tight loop (`r1_parity_check`, 2000 iters) | 23.6 µs | 24.7 µs | 43.5 µs | 387 µs |
| **50 Hz duty cycle (`r1_policy_node`)** | **194 µs** | **217 µs** | **378 µs** | 474 µs |

Same engine, same host. The control loop does ~24 µs of work every 20 ms, which
is far too little to pull the GPU out of its idle power state: `nvidia-smi`
reports `persistence_mode Disabled`, `pstate P8`, SM clock **450 MHz against a
2100 MHz maximum**. The tight loop keeps the GPU boosted and measures a
condition the real system never operates in.

Both numbers are comfortable against the 20 ms budget (the honest one is ~1%),
but quote the 50 Hz column. On Orin the equivalent knobs are `nvpmodel` and
`jetson_clocks`, and they should be set before latency is characterised there.

A corollary for W08: at batch 1 this model's latency is dominated by launch
overhead, not arithmetic — FP16 produced **bit-identical** output to FP32 here
because TensorRT chose FP32 kernels as faster. Expect INT8 quantisation to buy
memory and power, not milliseconds.

## Porting to the robot (W06)

The artefact that travels is `policy.onnx` plus `interface/policy_interface.*`.
Engines do not travel.

1. Copy `deploy/` to the Jetson, minus `.venv/`, `third_party/` and `artifacts/*.plan`.
2. Do **not** run `setup.sh` — JetPack supplies TensorRT and CUDA. `env.sh`
   detects the absence of `third_party/` and falls back to `/usr`.
3. `colcon build`, then `r1_build_engine` on the Jetson.
4. `r1_parity_check` with the same `parity_fixture.bin` copied over. **Gate.**
5. `check_env.py` on both hosts; diff the two outputs and record the TensorRT
   version skew.
6. `mode:=selftest` first, then `mode:=subscribe` against the real sensor bridge
   with the robot hanging.

## Observation layout — the one thing that will silently break

The 425 floats are **term-major**, not frame-major:

```
[ang_vel×5][gravity×5][command×5][joint_pos×5][joint_vel×5][action×5]
```

not five consecutive 85-float frames. Both are 425 floats long, so the wrong
one loads and runs and produces confident garbage. `ObsAssembler` owns this and
`test_obs_assembler.cpp` asserts the correct layout *and* explicitly asserts
inequality with the frame-major one. Publishers should send the current frame
only and let the node stack it.

The 24 action joints are in **articulation order**, which is the robot's
kinematic-tree order and not the order of the regexes that selected them —
`waist_roll_joint` sits at index 2, between the hip pitches and the hip rolls.
Read the order out of `interface/policy_interface.md`; do not retype it.
