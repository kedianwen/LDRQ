# R1process

Working directory for the R1 Sim2Real project: the Isaac Lab training side (R1
asset pipeline, manager-based task, verification scripts) and the on-robot
deployment side (ONNX → TensorRT → C++/ROS 2). Practical counterpart to the
12-week dev plan in `~/kdw/development plan/`.

**Status: W01–W07 closed. W08 (the M2 gate) in progress.** The policy walks on
the real robot under protection and recovers from being pushed; what is left for
M2 is walking off the gantry for 60 s (PG-2), INT8 quantisation and its
behavioural acceptance.

| Milestone | Weeks | Verdict |
|---|---|---|
| M1 · training | W01–W04 | **passed** 2026-08-19 — PG-1 met with 4x margin |
| M2 · deployment + quantisation | W05–W08 | in progress — real-robot walking achieved, PG-2 and INT8 open |
| M3 · robustness | W09–W12 | not started |

## Layout

- `assets/` — the R1 robot definition: URDF, meshes, Isaac Lab `ArticulationCfg`,
  and the USD build artifact. See `assets/README.md`.
- `tasks/` — the R1 flat-ground velocity-tracking gym task (manager-based:
  scene/actions/observations/rewards/terminations/events). See `tasks/README.md`.
- `scripts/` — tools that operate on the asset and task: URDF→USD conversion,
  load/limit/standing verification, task smoke test, GPU throughput sweeps.
  See `scripts/README.md`.
- `docs/` — dated verification reports, figures and video produced by the scripts
  above, kept as exit-criteria evidence (W01/W02 joint and standing checks, the
  M1 gate figures, the W06 walking clip). See `docs/README.md`.
- `experiments/` — git-tracked training-run records (config + final metrics)
  for comparing runs across ablations. See `experiments/README.md`.
- `deploy/` — the on-robot half: ONNX export, TensorRT engine builder and parity
  checker, the C++/ROS 2 hardware bridge and policy runner, and the Python probes
  that measure the robot. Self-contained and sudo-free. See `deploy/README.md`.

Week-by-week execution records and the on-robot step-by-step documents live
outside this repository, in `~/kdw/experiment_record/` and `~/kdw/`.


## What each week produced

Closed weeks, with the finding from each that was not visible in any metric.
That column is the useful one: on this project almost every real defect has been
silent — the chain reports success and returns a plausible wrong answer.

| Week | Delivered | What was not obvious |
|---|---|---|
| **W01** · environment + asset | URDF→USD conversion, `ArticulationCfg`, joint/limit verification, 2080 Ti throughput sweep. Closed 2026-08-12. | R1 is the **EDU** version with an onboard Orin NX, so deployment needs no external compute board — this set the whole M2 plan. |
| **W02** · task skeleton + standing | `tasks/r1_flat/` as a standalone package registering `Isaac-Velocity-Flat-R1-v0`, observations split policy/critic for asymmetric AC, static standing ≥10 s. Closed 2026-08-14. | Standing needed **`sim dt=0.002`**, not more gain tuning. At the default step the contact solve could not hold a 26-DoF biped, and that reads as a tuning problem. |
| **W03** · rewards + first training | First real alternating gait (`touchdown_gate`), plus `scripts/diagnose_gait.py` (per-foot air-time fraction + FFT phase). Closed 2026-08-15. | **`feet_air_time_positive_biped` is maximised by never stepping.** One foot planted and the other permanently airborne is that term's global optimum, so the policy learned single-leg support. Three reward rounds missed it; the diagnostic script found it in one. |
| **W04** · DR + M1 gate | Domain randomisation, observation history, hardware-spec actuators, command curriculum, symmetry augmentation; head not actuated → **24-dim action / 425-dim observation**. Final run `2026-08-19_11-03-32_week04_nohead`. **M1 passed** — PG-1 worst 0.037 m/s against a 0.15 threshold. | A config *dump* is weaker than a spec: it records what happened without checking the code still produces it. `scripts/verify_repro.py` turns the dump into a contract the repo is tested against. |
| **W05** · ONNX → TensorRT on the Orin | Engine builder, parity checker, C++/ROS 2 runner, running on the robot's own Orin NX. **FR-Q1 passed on target hardware: `max_abs = 1.335e-05`** against a 1e-3 gate; 50 Hz closed loop, `failures=0`, inference 2.45% of the control budget. | Numerics held across **three simultaneously different dimensions** — Turing→Ampere, TensorRT 10.7→8.5.2, ROS 2 humble→foxy — which is far stronger evidence than a dev-box pass. Separately, pinning the clocks tightened the latency tail **11x**, so any unpinned timing number is a lottery. |
| **W06** · C++/ROS 2 node + hanging dry run | Hardware bridge, watchdog, fault injection, 7/7 exit criteria. The robot walks under protection. | Two defects that produced **no error anywhere**: (1) **`--fp32` was never fp32** — TensorRT enables `kTF32` by default, rounding GEMM inputs to a 10-bit mantissa (FP16's width), and because the flag only *permits* TF32 the choice is made by build-time kernel timing, so the same command passed once at 1.144e-05 and failed later at 1.095e-02 on the same machine and ONNX. (2) **PD gains are per joint**, six groups, not one scalar — an export omission left the legs undamped and the head vibrating. |
| **W07** · walking on the real robot | Walking in developer mode with push recovery. Setpoint-lag instrumentation in the bridge, stance attribution tooling. | **Developer mode is a second writer problem.** Without switching the handheld first, the factory motion service keeps writing `rt/lowcmd` at 500 Hz against us; the symptoms — torso sway, joint grinding, tremor, stance not held — all look exactly like a sim2real gap, and cost a full session. Two measurements then *cancelled* planned work: setpoint lag is **1.3 ms = 0.065 control steps** against a trained range of {0,1} steps, so no delay compensation; and **FP16 buys nothing** (1.717e-05, FP32's order, and 171 µs, not faster) because at batch 1 this 90k-parameter MLP is kernel-launch bound and the builder keeps FP32 kernels. Narrow stance was attributed to the hardware side with saturation ruled out arithmetically — 6.45 N·m against a 60 N·m rating, 10.8%. |

**W08 (in progress)** — INT8 PTQ with a calibration set recorded off the walking
robot, three-layer behavioural acceptance, a FP32/FP16/INT8 benchmark, and the
PG-2 sprint. Two things already worth recording: the plan's closed-loop
no-regression gate is **not executable as literally written**, because a
TensorRT plan is not portable — the INT8 engine exists only on the Orin and the
simulator only on the dev box — so it runs against a proxy engine built from the
same calibration set, labelled as such. And INT8 is **not expected to be faster
or smaller** here; its value is closing FR-Q3's deployment loop and giving M3 a
controlled, physically real perturbation source (FR-R2/PG-5).

## Environment

Isaac Lab 2.1.0 + Isaac Sim 4.5.0 + torch 2.5.1+cu118, conda env `env_isaaclab`.
All scripts run through `~/IsaacLab/isaaclab.sh -p ...` from this directory (the
project root) — see each script's docstring/header for the exact invocation.

## Reproducing a training run (FR-T6)

Everything needed to reproduce a run is pinned: the seed is set explicitly in
`tasks/r1_flat/agents/rsl_rl_ppo_cfg.py` (not inherited from an rsl_rl default),
and Isaac Lab dumps the fully-resolved configs to
`logs/rsl_rl/r1_flat/<run_id>/params/{env,agent}.yaml` at launch. Each archived
run in `experiments/<run_id>/` keeps those configs plus a `pip freeze` snapshot.

**Where the hyperparameters actually live.** They are Python dataclasses
(`tasks/r1_flat/flat_env_cfg.py`, `.../agents/rsl_rl_ppo_cfg.py`), and the
`params/*.yaml` beside each run are a *dump* of the resolved config — editing
that yaml changes nothing. This is Isaac Lab's design, and it keeps type
checking and IDE navigation over the config, but on its own a dump is weaker
than a spec: it records what happened without anything checking that the code
still produces it.

`scripts/verify_repro.py` closes that. It rebuilds the config from current code
and diffs it field by field against a run's archived yaml, so the dump becomes a
contract the repository is tested against:

```bash
~/IsaacLab/isaaclab.sh -p scripts/verify_repro.py --headless \
    --run 2026-08-19_11-03-32_week04_nohead
# RESULT: PASS -- current code reproduces this run's configuration exactly
```

It exits non-zero on drift and writes `experiments/<run_id>/repro_check.txt`.
Fields that vary per invocation (env count, device, run name) are reported
separately from real drift, and values the scene resolves at construction time
(`{ENV_REGEX_NS}` placeholders, terrain env count/spacing) are normalised rather
than ignored. Run it against a superseded run to see it work: `week04_hwspec`
reports exactly the two fields the head removal changed.

```bash
conda activate env_isaaclab

# train (Week04 final config: hardware-spec actuators + obs history +
# domain randomization + command curriculum + symmetry augmentation, head not
# actuated -> 24-dim action, 425-dim policy observation).
# A single run -- the command curriculum counts env steps, so splitting this
# into train-then-resume shifts the ramp and does not reproduce the same curve.
~/IsaacLab/isaaclab.sh -p scripts/train_r1.py --headless \
    --num_envs 4096 --max_iterations 3000 --run_name week04_nohead

# watch it
tensorboard --logdir logs/rsl_rl/r1_flat

# evaluate: PG-1 tracking table, then per-foot gait statistics
~/IsaacLab/isaaclab.sh -p scripts/eval_baseline.py --headless --num_envs 64 \
    --checkpoint logs/rsl_rl/r1_flat/<run_id>/model_2999.pt
~/IsaacLab/isaaclab.sh -p scripts/diagnose_gait.py --headless --num_envs 32 \
    --checkpoint logs/rsl_rl/r1_flat/<run_id>/model_2999.pt

# archive config + final metrics into experiments/
python scripts/archive_run.py <run_id>
```

`--resume` re-interprets `--max_iterations` as an *absolute* target rather than
an additional count — see `scripts/train_r1.py`'s docstring.

## Hardware target

R1 is the **EDU version** with an onboard 8-core Jetson Orin NX (40-100 TOPS), so
the policy runs on the robot itself with no external compute board.

The robot's software stack is **not** this dev box's, and every difference has
bitten at least once: JetPack 5.1.1, **TensorRT 8.5.2**, **ROS 2 foxy**, CUDA
11.4, and `unitree_hg` messages. A TensorRT `.plan` is tied to the TensorRT
version, GPU architecture and driver that built it, so **the artefact that ships
to the robot is the ONNX, and the engine is built on the Orin**. `deploy/` keeps
both toolchains working from one tree; see `deploy/README.md`.
