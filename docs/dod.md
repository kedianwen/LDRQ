# Definition of done — checklist (2026-09-29)

The project's closing checklist, written at the end of the 12-week plan against the
project's requirements document. That document and the week plans are not published, so
each gate (PG-n) and requirement (FR-xn) below is restated in its own row, with the
evidence that closes it; the ID scheme is explained in the top-level README's
[*How the project was organised*](../README.md#how-the-project-was-organised). **Status legend:** ✅ met · ⚖️ waived or cut, with evidence ·
⏳ met, evidence being archived · ◻ open.

## Gates

| gate | what it asks | status | evidence |
|---|---|---|---|
| **PG-1** | velocity tracking ≤ 0.15 m/s over 0.5–1.0 m/s (sim) | ✅ | worst 0.037 m/s, 0 falls — `scripts/eval_baseline.py`, [m1_dr_robustness.png](m1_dr_robustness.png) |
| **PG-2** | 60 s continuous walking on the real robot (or the fallback line) | ✅ ⏳ | met at the target line: ≥ 79.7 s untethered, RUNNING throughout, no DEGRADED, recorded 2026-09-29 — [pg2_untethered.md](pg2_untethered.md). The video is deferred to the end of the project (space and a second person) |
| **PG-3** | the inference chain verified on the target hardware | ✅ | parity 1.335e-05 on the Orin (gate 1e-3); in-loop inference p50 ≈ 455 µs of 20 ms; 50.0 Hz, `failures=0` — [technical_report.md §2](technical_report.md#2-simulation-and-deployment-results) |
| **PG-4** | precision tiers benchmarked; deployment loop closed | ✅ ⚖️ | FP32 and FP16 measured on the Orin; INT8 waived on three measurements — [int8_waiver.md](int8_waiver.md). Train → ONNX → Orin engine → parity → C++/ROS 2 → robot closed since W06 |
| **PG-5** | robustness experiment done and plotted (FR-R2) | ✅ | `kp_scale` stability domain, 16 gains × 64 envs in sim — [stageA_kp_sweep.md](stageA_kp_sweep.md) |
| **PG-6** | sim2real gap quantified (FR-R4) | ✅ | same figure: real 1.10–1.50 vs sim 0.80–2.00+; gap +0.30 / ≤ −0.50, attributed to posture, not joint tracking |
| **PG-7** | repository reproducible by a third party | ✅ | see *Clean-copy check* below |

## Functional requirements

| ID | what | status | evidence |
|---|---|---|---|
| FR-T1 – T5 | asset, task, rewards, training, five-item DR | ✅ | W01–W04; `tasks/r1_flat/`, `experiments/` |
| FR-T6 | reproducible training config | ✅ | `scripts/verify_repro.py` PASS on 2026-09-29 against the deployed run |
| FR-Q1 | numerical parity of the deployed engine | ✅ | 1.335e-05 on the Orin |
| FR-Q2 | on-robot latency | ✅ | [int8_waiver.md](int8_waiver.md) §1, technical report §2 |
| FR-Q3 | INT8 PTQ | ⚖️ waived | [int8_waiver.md](int8_waiver.md) |
| FR-Q4 | precision-tier benchmark | ✅ at FP32 / FP16 | [int8_waiver.md](int8_waiver.md) |
| FR-D* | C++/ROS 2 bridge, watchdog, fault injection | ✅ | W06 7/7; `deploy/README.md` |
| FR-R2 | robustness under a controlled physical perturbation | ✅ | knob changed from INT8 to `kp_scale`, methodology unchanged |
| FR-R3 | DR ablation | ⚖️ cut | cut for time (2026-09-26); partly answered for the stiffness item — technical report §6 |
| FR-R4 | sim2real gap | ✅ | [stageA_kp_sweep.md](stageA_kp_sweep.md) |

## Deliverables

| deliverable | status | where |
|---|---|---|
| technical report | ✅ (Markdown) | [technical_report.md](technical_report.md). No PDF built: this machine has no pandoc/LaTeX |
| public, reproducible repository | ✅ | this repository; *Reproducing* in the top-level README |
| demo video | ◻ one clip missing | `scripts/record_demo_sim.py` + `scripts/make_demo_video.py` assemble title → simulation (same sequence as the robot) → robot walking and recovering from pushes (`w07_walk_push_recovery.mp4`) → stage A figure → summary. The draft exists; the final cut adds the untethered clip, recorded last |

## Clean-copy check (PG-7)

The tree that would be committed was exported to an empty directory (no `logs/`, no
`deploy/artifacts/`, no USD) on 2026-09-29 and each "none"-row of README *Reproducing* was
run there. Since then the same checks run on every push in CI from a fresh clone
(`./run_tests.sh`; the badge on the top-level README); test counts are in its output,
not copied here.

- `models/week04_nohead/SHA256SUMS`: all files match
- `mission_ctl/tests/test_core.py`: all pass
- `policy_pack/tests/run_tests.sh`: all pass — **failed before this check** (it needed the
  untracked `deploy/artifacts/`); `make_bundle.py` now falls back to `models/`
- stage A figure redrawn from the committed JSON: byte-identical to `docs/stageA_kp_sweep.png`
- the GPU path: the simulated sweep **failed before this check** — the robot's USD is a
  gitignored build artefact and the top-level README did not say to generate it first
  (now step 0 of *Reproducing*). After `convert_r1_urdf.py` (26 joints) the sweep ran from
  the clean copy with the policy from `models/` and reproduced **all 16 points exactly**,
  same stable domain.

Not done: a check on a second machine or by a second person, which the closing plan
recommended (CI now covers the second machine for the no-GPU checks). Everything above ran on the development box, in a directory with no access to
the untracked files.
