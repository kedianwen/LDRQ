# Project history: plan, gates, and what each week and stage found

The top-level [README](../README.md) describes the result. This page is the process
behind it, for a reader who wants to know how the work was organized, what "done" was
defined as, and what each step found that no metric showed.

## How the project was organized

The work followed a 12-week plan, **W01–W12**, in three milestones, **M1–M3**. On
2026-09-26, after W08, the last four weeks were replaced by four **stages, A–D**, because
dropping INT8 quantization (see below) made the original W09–W12 describe a different
project. Code, commits and documents refer to work by week or stage:

| period | milestone | what was built | where it lives |
|---|---|---|---|
| **W01** (closed 2026-08-12) | M1 training | robot model: URDF → USD, Isaac Lab config, joint and mass checks, GPU throughput | `assets/`, `scripts/` |
| **W02** (08-14) | M1 | the gym task, standing still | `tasks/r1_flat/` |
| **W03** (08-15) | M1 | rewards, first real gait, gait diagnostics | `tasks/`, `scripts/diagnose_gait.py`, `experiments/` |
| **W04** (08-19) | M1 | domain randomization, observation history, hardware-spec actuators; **the deployed policy** | `models/week04_nohead/`, `experiments/` |
| **W05** (08-26) | M2 deployment | ONNX → TensorRT on the Orin, numerical parity | `deploy/` |
| **W06** (09-23) | M2 | C++/ROS 2 hardware bridge, watchdog, first walk under a gantry | `deploy/ros2_ws/` |
| **W07** (09-24) | M2 | walking and push recovery on the real robot; measurement tools | `deploy/tools/` |
| **W08** (09-28) | M2 | INT8 evaluated and dropped; 60 s untethered walk | [docs/int8_waiver.md](int8_waiver.md) |
| **Stage A** (09-29) | M3 robustness | the `kp_scale` sweep, sim and real on one axis; report; reproducible repo | [docs/stageA_kp_sweep.md](stageA_kp_sweep.md), [docs/technical_report.md](technical_report.md) |
| **Stage B** (09-30) | M3 command layer | `mission_ctl`: walk by time, turn by angle, abort paths, on the robot | `mission_ctl/` |
| **Stage C** (09-30) | M3 | English instructions through a local model on the Orin | `mission_ctl/r1_mission/nl.py`, `llm/` |
| **Stage D** (10-02) | M3 | swap the policy with one command, accepted on the robot | `policy_pack/`, [docs/stageD_policy_swap.md](stageD_policy_swap.md) |

The plans themselves, the lab notebook and the step-by-step guides used at the robot
are kept outside this repository. Everything a reader needs from them is restated here
and in `docs/`. Three traces of them remain:

- **Requirement IDs** such as PG-2 or FR-Q3. They are explained below, in [*What "done" meant*](#what-done-meant-gates-and-requirements). **"Plan 3.6"**
  is the stage C plan's
  coexistence criterion: while the language model runs on the shared GPU, policy
  inference p99 ≤ 2 ms and setpoint lag p95 ≤ idle + 1 ms
  ([docs/system_architecture.md](system_architecture.md#the-shared-gpu-decision-2026-09-30)).
- **`~/kdw/...` paths** in a few code comments point at that private notebook. The
  finding each one cites is summarized next to it.
- **`experiments/*/NOTES.md`** are the original lab notes, in Chinese. Each run is
  summarized in English in [experiments/README.md](../experiments/README.md), and the
  lessons are in [*What each week produced*](#what-each-week-produced-and-what-was-not-obvious) below.

## What "done" meant: gates and requirements

The plan defined success up front with three kinds of ID, and the documents use them as
shorthand. Every one that appears in this repository:

- **M1–M3** are the three milestones.
- **PG-n** ("project gate") is a pass/fail test a milestone has to meet.
- **FR-xn** ("functional requirement") is something the system must do. The letter says
  which part: **T** training, **Q** quantization and inference, **D** deployment on the
  robot, **R** robustness.
- **RK-n** is a risk the plan asked to check early.

| ID | in plain words | outcome |
|---|---|---|
| **M1** | **Training** (W01–W04): a policy that walks in simulation and can be deployed as is | passed 2026-08-19 |
| **M2** | **Deployment** (W05–W08): that policy walking on the real robot, on its own computer | met; INT8 waived |
| **M3** | **Robustness and command layer** (stages A–D): one measured sim2real experiment, a way to command the robot, report and repository | A–D done (videos at the end) |
| PG-1 | in simulation, forward-speed tracking error ≤ 0.15 m/s anywhere in 0.5–1.0 m/s | ✅ worst 0.037 m/s, no falls |
| PG-2 | on the real robot, 60 s of continuous walking (a weaker fallback was allowed: walking in the gantry plus a gap analysis) | ✅ ≥ 79.7 s untethered |
| PG-3 | the inference chain runs on the robot's own Orin and matches the PyTorch reference | ✅ max difference 1.3e-5, inference ~0.5 ms of a 20 ms step |
| PG-4 | precision options (FP32 / FP16 / INT8) measured on the robot, and the train → robot loop closed | ✅ FP32 and FP16; INT8 waived (FR-Q3) |
| PG-5 | the robustness experiment run and plotted | ✅ the `kp_scale` stability domain |
| PG-6 | the sim2real gap put in numbers | ✅ real 1.10–1.50 against sim 0.80–2.00+, on the same figure |
| PG-7 | a third party can reproduce the repository | ✅ clean-copy check, and CI on every push |
| FR-T1 – T3 | robot model in Isaac Lab, the training task, the reward function | ✅ W01–W03 |
| FR-T4 | the policy uses only what the robot can sense, so it deploys without a teacher-student distillation step (done with a 5-frame observation history) | ✅ W04 |
| FR-T5 | five-item domain randomization: friction, mass, motor strength, control delay, pushes | ✅ W04 |
| FR-T6 | a training run can be reproduced exactly: fixed seed, every setting in config | ✅ `scripts/verify_repro.py` |
| FR-Q1 | the exported model (ONNX → TensorRT) gives the same numbers as PyTorch | ✅ on the Orin |
| FR-Q2 | inference latency measured on the robot | ✅ |
| FR-Q3 | INT8 post-training quantization, with a calibration set | ⚖️ **waived**: no speed gain, a larger engine, no defensible calibration data ([docs/int8_waiver.md](int8_waiver.md)) |
| FR-Q4 | latency, size and power compared across precisions | ✅ at FP32 and FP16 |
| FR-D* | the C++/ROS 2 control node: 50 Hz policy over a 500 Hz motor loop, Unitree SDK integration, watchdog and fault injection | ✅ W06 |
| FR-R2 | **the headline experiment**: sweep one controlled, physically real disturbance in simulation, spot-check it on the robot, draw the stability domain. Planned with INT8 as the disturbance; done with actuator stiffness `kp_scale` | ✅ stage A |
| FR-R3 | domain-randomization ablation: which randomization item buys how much robustness | ⚖️ cut for time |
| FR-R4 | quantify the sim2real gap | ✅ the offset between the sim curve and the real points of FR-R2 |
| RK-7 | is a single RTX 2080 Ti fast enough to train on? | ✅ answered in W01 ([docs/throughput_sweep.md](throughput_sweep.md)) |

The checklist with the evidence for each row is [docs/dod.md](dod.md).

## Status by stage

**The 12-week scope and all four stages are closed (stage D on 2026-10-02).** All videos
are recorded at the end of the project.

| Milestone | Span | Verdict |
|---|---|---|
| M1 · training | W01–W04 | **passed** 2026-08-19 — speed tracking (PG-1) met with 4x margin |
| M2 · deployment | W05–W08 | **met** — 60 s untethered walk (PG-2; video being archived), same numbers as PyTorch on the Orin to 1.3e-5 (FR-Q1), INT8 waived with evidence (FR-Q3), precisions compared at FP32/FP16 (FR-Q4) |
| M3 · robustness + command layer | stages A–D | **A, B, C, D done** (videos at the end) |

### Stage A: robustness: the `kp_scale` stability domain

**Goal.** `kp_scale` stability domain (sim curve + real points on one axis, same command sequence on both sides), evidence for the 60 s walk (PG-2), report/repo/video/DoD. The speed calibration was dropped on 2026-09-28: ground speed is taken as equal to the command

**Outcome.** **done** except the PG-2 video: sim 0.80–2.00+ vs real **1.10–1.50** (both edges bounded: the trained 1.00 passed once and failed once on an 80 ms tilt spike; 1.60 was emergency-stopped on audible joint noise). Report, INT8 waiver, DoD in `docs/`; demo assembled except the PG-2 clip

### Stage B: the command layer on the robot

**Goal.** `mission_ctl/` on the robot: walk for a time, turn to an angle, walk an (open-loop) distance at the commanded speed. Demo at `kp_scale` 1.2 / 1.3, the middle of the real domain

**Outcome.** **done** 2026-09-30 except the video. Turn response on the spot at kp 1.0/1.2/1.3: every rate 0.15–0.5 rad/s turns at ~0.8 of the command; a small yaw rate is lost *while walking*, so the robot turns on the spot only ([docs/stageB_turn_response.md](stageB_turn_response.md)). At kp 1.3: 10 closed-loop turns and the demo sequence all DONE, 0 timeouts, within 1.6° by the IMU; abort paths 4/4 (Ctrl-C, `kill -9` → deadman, DEGRADED, no stack) ([docs/stageB_mission_runs.md](stageB_mission_runs.md)). By decision, turn angle is the IMU reading and speed is the command

### Stage C: English instructions through a local model

**Goal.** LLM command layer: **English** instruction → schema-constrained JSON → deterministic checks → execution → templated report

**Outcome.** **done** 2026-09-30 except the video. `mission_ctl ask`: the model transcribes, deterministic code judges (units normalized first; every distance, time and angle must have been said; refusals from the deployed envelope; all or nothing), the operator confirms the parsed plan, the reply is a template. Offline eval of 7 small models on three held-out sets frozen in turn: **qwen3:1.7b 75/80 clean**, 100 % valid JSON. **On the Orin** (Ollama 0.34.4 for JetPack 5): the same 77/80 as the dev box, 0.65 s p50; 11 instructions end to end, all executed or refused as asked. The model stays on the GPU by decision: it holds the policy's inference at ~5 ms, inside every control-loop limit but above plan 3.6's 2 ms target, which is the optimization goal. **Each `ask` stands alone** ([docs/stageC_nl_eval.md](stageC_nl_eval.md))

### Stage D: swapping the policy with one command

**Goal.** `policy_pack/` on the robot: swap a policy with one command

**Outcome.** **done** 2026-10-02. Regression install of the running policy: 8/8 steps, parity 1.144e-05 before and after, only the expected four files changed (`bridge.yaml` and the joint map byte-identical), same in-loop inference time (p50 ~454 µs, clocks locked); three bad bundles refused at step 1 with the deploy tree untouched; the stack walked on the new engine. The plan's "engine fingerprint reproduces" criterion was wrong — TensorRT rebuilds are not byte-identical — and parity is the gate ([docs/stageD_policy_swap.md](stageD_policy_swap.md))

## What each week produced, and what was not obvious

Closed weeks, with the finding from each that was not visible in any metric.
That column is the useful one: on this project almost every real defect has been
silent — the chain reports success and returns a plausible wrong answer.

### W01 · environment + asset

**Delivered.** URDF→USD conversion, `ArticulationCfg`, joint/limit verification, 2080 Ti throughput sweep. Closed 2026-08-12.

**Not obvious.** R1 is the **EDU** version with an onboard Orin NX, so deployment needs no external compute board — this set the whole M2 plan.

### W02 · task skeleton + standing

**Delivered.** `tasks/r1_flat/` as a standalone package registering `Isaac-Velocity-Flat-R1-v0`, observations split policy/critic for asymmetric AC, static standing ≥10 s. Closed 2026-08-14.

**Not obvious.** Standing needed **`sim dt=0.002`**, not more gain tuning. At the default step the contact solve could not hold a 26-DoF biped, and that reads as a tuning problem. (The real cause was gains far stiffer than the hardware's; after W04 corrected them, `dt=0.005` was stable again.)

### W03 · rewards + first training

**Delivered.** First real alternating gait (`touchdown_gate`), plus `scripts/diagnose_gait.py` (per-foot air-time fraction + FFT phase). Closed 2026-08-15.

**Not obvious.** **`feet_air_time_positive_biped` is maximized by never stepping.** One foot planted and the other permanently airborne is that term's global optimum, so the policy learned single-leg support. Three reward rounds missed it; the diagnostic script found it in one.

### W04 · DR + M1 gate

**Delivered.** Domain randomization, observation history, hardware-spec actuators, command curriculum, symmetry augmentation; head not actuated → **24-dim action / 425-dim observation**. Final run `2026-08-19_11-03-32_week04_nohead`. **M1 passed** — speed-tracking error (PG-1) worst 0.037 m/s against a 0.15 threshold.

**Not obvious.** A config *dump* is weaker than a spec: it records what happened without checking the code still produces it. `scripts/verify_repro.py` turns the dump into a contract the repo is tested against.

### W05 · ONNX → TensorRT on the Orin

**Delivered.** Engine builder, parity checker, C++/ROS 2 runner, running on the robot's own Orin NX. **Numerical parity with PyTorch (FR-Q1) passed on target hardware: `max_abs = 1.335e-05`** against a 1e-3 gate; 50 Hz closed loop, `failures=0`, inference 2.45% of the control budget.

**Not obvious.** Numerics held across **three simultaneously different dimensions** — Turing→Ampere, TensorRT 10.7→8.5.2, ROS 2 humble→foxy — which is far stronger evidence than a dev-box pass. Separately, pinning the clocks tightened the latency tail **11x**, so any unpinned timing number is a lottery.

### W06 · C++/ROS 2 node + hanging dry run

**Delivered.** Hardware bridge, watchdog, fault injection, 7/7 exit criteria. The robot walks under protection.

**Not obvious.** Two defects that produced **no error anywhere**: (1) **`--fp32` was never fp32** — TensorRT enables `kTF32` by default, rounding GEMM inputs to a 10-bit mantissa (FP16's width), and because the flag only *permits* TF32 the choice is made by build-time kernel timing, so the same command passed once at 1.144e-05 and failed later at 1.095e-02 on the same machine and ONNX. (2) **PD gains are per joint**, six groups, not one scalar — an export omission left the legs undamped and the head vibrating.

### W07 · walking on the real robot

**Delivered.** Walking in developer mode with push recovery. Setpoint-lag instrumentation in the bridge, stance attribution tooling.

**Not obvious.** **Developer mode is a second writer problem.** Without switching the handheld first, the factory motion service keeps writing `rt/lowcmd` at 500 Hz against us; the symptoms — torso sway, joint grinding, tremor, stance not held — all look exactly like a sim2real gap, and cost a full session. Two measurements then *cancelled* planned work: setpoint lag is **1.3 ms = 0.065 control steps** against a trained range of {0,1} steps, so no delay compensation; and **FP16 buys nothing** (1.717e-05, FP32's order, and 171 µs, not faster) because at batch 1 this 90k-parameter MLP is kernel-launch bound and the builder keeps FP32 kernels. Narrow stance was attributed to the hardware side with saturation ruled out arithmetically — 6.45 N·m against a 60 N·m rating, 10.8%.

### W08 · and why INT8 was cut

 The week was planned around INT8
PTQ, a three-layer behavioral acceptance and a FP32/FP16/INT8 benchmark. It was
closed as a *conclusion* rather than as unfinished work, on three
independent measurements:

1. **It cannot buy time.** This policy is 90,648 parameters at batch 1, so a step
   is ~180 kFLOP — a fraction of a microsecond of arithmetic — against 70 µs measured
   on the Orin: over 99% of a step is kernel launch overhead. FP16 measured 1.717e-05 (FP32's order) and 171 µs (not
   faster), because `kFP16`/`kINT8` *permit* rather than require, and TensorRT
   picked FP32 kernels by build-time timing.
2. **It cannot buy space.** FP16 made the engine **51.1% larger**
   (966,887 → 1,461,299 B): the arithmetic saving does not cover the extra
   reformat layers and duplicated weights.
3. **The calibration set is not defensible.** The observation is five stacked
   frames (80% overlap between neighbours), so 60 s of single-speed straight
   walking is one operating point and ~72 gait cycles; covering the vx×wz grid
   needs several real walking segments, which did not exist when the decision
   was made.

So INT8 (FR-Q3) is recorded as *waived with evidence*, and the precision comparison
(FR-Q4) as *satisfied at two precisions*. A related finding kept from the same week: the plan's closed-loop
gate was **not executable as literally written**, because a TensorRT plan is not
portable — the engine exists only on the Orin and the simulator only on the dev
box.

M3 keeps the headline experiment's (FR-R2's) methodology — one controlled, physically real perturbation,
swept in simulation and spot-checked on the robot — and changes the knob to
**`kp_scale`**, the actuator-gain error. That knob is already a launch argument
on the robot, it is one of the five domain-randomization items, and unlike INT8
it actually moves the metrics. Because the same knob sweeps on both sides, one
figure settles the robustness experiment (the curve; FR-R2, PG-5) and the sim2real gap
(the offset between the curve and the real points; FR-R4, PG-6) at once.
