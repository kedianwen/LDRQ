# deploy/ — TensorRT + ROS 2 runner for the R1 policy

The on-robot half of the project, built in W05–W08 of the project plan (W05 ONNX →
TensorRT on the Orin, W06 the C++/ROS 2 hardware bridge, W07 walking on the real robot,
W08 precision tiers; see *How the project was organised* in the
[top-level README](../README.md#how-the-project-was-organised)). Self-contained:
nothing here needs sudo, and the training environment (`conda env_isaaclab`) is
never touched.

The same tree builds on two machines: the dev box (x86, TensorRT 10.7, ROS 2 humble),
where the *Quick start* below runs the policy with no robot, and the robot's Orin NX
(JetPack 5.1.1, TensorRT 8.5.2, ROS 2 foxy), where *Porting to the robot* applies.

At run time on the robot, two processes form the control loop:

```
rt/lowstate --> r1_hw_bridge --~/obs (50 Hz)--> r1_policy_node (TensorRT)
rt/lowcmd  <--  (PD, 500 Hz) <--~/joint_target--
                     ^ ~/cmd_vel  [vx, vy, wz] from mission_ctl (../mission_ctl/)
```

The policy it runs is `2026-08-19_11-03-32_week04_nohead` — 425-dim
observation, 24-dim action, 50 Hz.

## Quick start

```bash
# the deployed policy ships in git under models/; deploy/artifacts/ is the working area
mkdir -p deploy/artifacts && cp models/week04_nohead/{policy.onnx,policy.pt,parity_fixture.bin} deploy/artifacts/

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
| `tools/dump_actuator_gains.py` | resolves the six training actuator groups onto the 26 joints and writes `interface/actuator_gains.json`; ast-parses the asset rather than importing isaaclab, so it runs on the robot too |
| `tools/gen_obs_table.py` | cross-checks the four files that decide the observation (training export, `joint_map.hpp`, both node yamls) and writes `interface/obs_consistency_table.md`; **non-zero exit on any disagreement** |
| `ros2_ws/src/r1_policy_runner/` | the ROS 2 package: engine build, parity check, node |
| `interface/policy_interface.{json,md}` | generated deployment contract (tracked) |
| `artifacts/` | ONNX, fixture, engines (git-ignored) |

Robot-side measurement and commissioning tools (W07 onward). All read the
bridge's topics or files; none of them writes `rt/lowcmd`.

| path | role |
|---|---|
| `tools/w08_preflight.sh` | pre-run checklist: clock lock, second writer on `rt/lowcmd`, stale install, ROS env mismatch across terminals; `--live` diagnoses an empty topic graph. Exit code = FAIL count |
| `tools/probe_cpp/probe_lowcmd` | read-only `rt/lowcmd` subscriber: is anyone else writing? (developer mode does **not** stop `master_service`, so traffic is the criterion, not process names) |
| `tools/probe_cpp/probe_lowstate` | read-only `rt/lowstate` probe; `--map` is how the slot map was measured |
| `tools/probe_cpp/walk_metrics.py` | records one walking segment and reports survival, tracking, measured torque, smoothness. Refuses to report achieved speed without a tape measurement |
| `tools/probe_cpp/gain_sweep_real.py` | stage A: plans the `kp_scale` points, records each one while `mission_ctl` walks the shared sequence (window sized for the worst case, mission report saved beside it), and reduces the recordings -- scored over the commanded window only -- into the real-robot stability-domain interval, marking each edge as *bounded* (a point beyond it failed) or *not bounded* (testing stopped there). A point the operator stopped with no usable recording goes into `operator_log.txt` as FAIL (FAIL only: it can close an edge, never widen the domain) |
| `tools/probe_cpp/vcal.py` | tape + stopwatch readings → calibrated speed with a measured error bar. **Not run**: on 2026-09-28 the speed test was dropped (no venue; the task needs distance only to within "a few metres"), and `mission_ctl` assumes ground speed = command. Kept for if a measurement is ever made |
| `tools/record_calib_obs.py`, `tools/bench_precision.py` | INT8 calibration recorder and FP32/FP16/INT8 benchmark. **Kept but unused**: INT8 was cut on 2026-09-26 (see below); `r1_build_engine --int8` still works and is the evidence path for that decision |

Three executables:

- **`r1_build_engine`** — ONNX → serialised engine. Must run on the machine that
  will execute it.
- **`r1_parity_check`** — replays the fixture through the engine, compares
  against PyTorch, and profiles latency. **This is the gate**, not a nicety.
- **`r1_policy_node`** — the ROS 2 node. `mode:=subscribe` reads one 85-float
  sensor frame per cycle from `~/obs`; `mode:=selftest` self-drives at 50 Hz
  with no robot attached (bench rehearsal for W06's hanging dry-run).

## What sits on top of this

`deploy/` is the runtime. Two sibling directories package it, and neither
requires a change here:

- **`../policy_pack/`** — turns "swap the trained policy" into one command. It
  builds a bundle (ONNX + `policy_interface.json` + `actuator_gains.json` +
  `command_envelope.json` + provenance + its own parity fixture), then on the
  robot regenerates `joint_map.hpp` and both node yamls from it, rebuilds, builds
  the engine and gates on parity against the bundle's own fixture. The bundle
  deliberately does **not** carry the unitree_hg slot map: that is a property of
  the robot, established by measurement.
- **`../mission_ctl/`** — drives the robot by time, angle and speed using the
  topics the bridge already has (`~/cmd_vel` out, `~/imu` and `~/status` in).
  Turning is closed-loop on the IMU heading; distance is open-loop
  time × the commanded speed, and is labelled "not measured", because there is
  no base linear velocity in the observation and no odometry topic. Its `ask`
  command adds English instructions through a local model served from `../llm/`.

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

A corollary that ended up deciding W08: at batch 1 this model's latency is
dominated by launch overhead, not arithmetic — FP16 produced **bit-identical**
output to FP32 here because TensorRT chose FP32 kernels as faster. On the Orin
that held (1.717e-05, 171 µs — not faster) **and** the FP16 engine came out
**51.1% larger** (966,887 → 1,461,299 B), because the arithmetic saving does not
cover the extra reformat layers and duplicated weights.

**So INT8 was cut on 2026-09-26** rather than pursued: it cannot buy time, there
is no evidence it buys space, and a defensible calibration set needs several real
walking segments that do not exist yet (the observation is five stacked frames
with 80% overlap, so 60 s of single-speed walking is one operating point). FR-Q3
is waived with evidence; FR-Q4 is satisfied at two precisions. M3's controlled
perturbation is `kp_scale` — the actuator-gain error, already a launch argument
on the bridge — instead of numeric precision. See the repo README,
[`docs/int8_waiver.md`](../docs/int8_waiver.md) and
[`docs/stageA_kp_sweep.md`](../docs/stageA_kp_sweep.md).

## Porting to the robot (W06)

The artefact that travels is `policy.onnx` plus `interface/policy_interface.*`.
Engines do not travel.

1. Copy `deploy/` to the Jetson, minus `.venv/`, `third_party/` and `artifacts/*.plan`,
   as a `.tar.gz` (a zip loses the execute bits). The tree that was copied in W07,
   Orin build output included, is a release asset: `bash tools/fetch_orin_snapshot.sh --extract`
   puts it at `~/kdw_deploy`.
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

## The hardware bridge (`r1_hw_bridge`)

Added in W06. The only package that knows Unitree message types:

```
rt/lowstate (DDS)  ->  ~/obs            85 floats per control step
~/joint_target     ->  rt/lowcmd (DDS)  24 targets -> 35 motor slots
```

The policy node speaks `Float32MultiArray` and is unaware of any of this, which
is what lets it be unit-tested, replayed from a rosbag, and driven from
simulation with no SDK present.

**Output is off by default.** `enable_output:=false` (the default) still reads
the robot, publishes `~/obs`, and computes every command onto `~/cmd_debug` --
so the whole chain, joint mapping included, can be checked against a powered
robot that cannot move. Turning it on is a separate deliberate act.

Two rates: observations at 50 Hz (the rate the policy trained at) and commands
at 500 Hz on a dedicated thread, so a slow inference tick cannot stall the motor
bus. Targets are re-sent unchanged between policy ticks rather than
interpolated -- in training the target was a 20 ms step, and smoothing it here
would be a sim-to-real difference introduced by deployment code.

### Why the package is split in two

`robot_io.cpp` is the only file that includes a Unitree header, and it is
compiled with **no ROS on its include path**. The node is compiled with **no
`/usr/local/include` on its include path**. They meet through the POD types in
`robot_io.hpp`.

This is not tidiness. Unitree installs CycloneDDS **0.10.2** under
`/usr/local/include/dds/`; ROS foxy installs **0.7.0** under
`/opt/ros/foxy/include/dds/`. A target that sees both prefixes resolves each
header from whichever comes first, so the compile picks up a blend of the two
versions -- `ddsi_sertype does not name a type` and a dozen more, thrown from
inside the SDK's own headers. Reordering the include paths makes the errors go
away by luck; the split makes the situation impossible. `CMakeLists.txt` filters
the offending prefix out of each target's `INCLUDE_DIRECTORIES` after the fact,
so a vendor CMake config that injects a directory-scope `include_directories()`
cannot reintroduce it.

The same version clash is why `RMW_IMPLEMENTATION` is pinned to
`rmw_fastrtps_cpp` in `env.sh`: at runtime, loading `rmw_cyclonedds_cpp` would
put both CycloneDDS versions in one address space.

### Joint map

`include/r1_hw_bridge/joint_map.hpp` is **generated** by
`tools/probe_cpp/gen_joint_map.py` and must never be hand-edited. It combines
three sources that are deliberately kept separate so a disagreement is visible:
our `policy_interface.json`, the vendor's `R1JointIndex` enum mapped through its
motor-index array, and `probe_lowstate --map` measurements taken on the robot.
Evidence per row is in `interface/joint_map_r1.md`.

Do not `#include` the vendor enum: it has `RightShoulderPitch = 29`, a typo for
19, which used as a slot number addresses `head_pitch`.

### Degrade behaviour

`/usr/local/include/unitree/robot/` ships clients for a2, b2, g1, go2 and h1 --
**not r1**. There is no vendor damping or e-stop call to delegate to, so the
bridge implements it: `kp=0, kd=damping_kd, tau=0`. It enters that state on a
stale `rt/lowstate`, a stale or non-finite `~/joint_target`, or three
consecutive one-second windows below `min_control_rate_hz`.

Recovery is explicit: publish `true` to `~/resume`. That makes the bridge send
`~/policy_reset`, which clears the policy's 5-frame history and re-warms it. An
automatic restart would feed the policy a history straddling the outage, which
is outside its training distribution.

**Numeric launch arguments must be written as floats.** `kp_scale`, `kd_scale` and
`min_control_rate_hz` are declared as `double`. The launch file passes the command-line
text through as YAML, so `min_control_rate_hz:=55` arrives as an integer. rclcpp then
throws `InvalidParameterTypeException` and the bridge dies at start-up, while the policy
node keeps running. Write `55.0`, `1.0`, `1.3`. This was found on the robot on
2026-09-30, when the DEGRADED abort test never reached DEGRADED. The launch file is left
as it is: changing it means a rebuild on the robot, and the rule is enough.

### Protocol requirements, all mandatory

| | |
|---|---|
| `mode_pr` | must be `PR` (0). `AB` addresses the ankles' parallel actuators instead of their pitch/roll joint angles |
| `mode_machine` | echoed back from `rt/lowstate` on every command |
| `crc` | recomputed per frame with the vendor's non-standard CRC32 (`crc32.hpp`); incoming `LowState` is checked too |
