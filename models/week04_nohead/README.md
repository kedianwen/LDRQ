# The deployed policy: `2026-08-19_11-03-32_week04_nohead`

The policy that walks on the robot, kept in git so that a fresh clone can evaluate,
deploy and re-run every figure without retraining (training logs under `logs/` are not
tracked: checkpoints and tensorboard events for all runs are several GB).

| file | what it is | used by |
|---|---|---|
| `model_2999.pt` | rsl_rl checkpoint, final iteration (actor + critic + optimiser) | `scripts/eval_baseline.py`, `scripts/diagnose_gait.py`, `scripts/play_r1.py` (`--checkpoint`) |
| `policy.pt` | TorchScript actor exported from that checkpoint: 425-dim observation → 24-dim action | `scripts/sweep_gain_robustness.py`, `deploy/tools/make_fixture.py` |
| `policy.onnx` | the same actor as ONNX — **the artefact that ships to the robot** | `deploy/` (`r1_build_engine`), `policy_pack/make_bundle.py` |
| `parity_fixture.bin` | 512 observation vectors and this policy's reference outputs | `r1_parity_check` on the robot (passed at 1.335e-05), and the sim sweep's start-up check |

`SHA256SUMS` lists all four. `policy.onnx` and `policy.pt` are byte-identical to
`logs/.../exported/` of the run and to what was installed on the robot.

No TensorRT engine is kept here on purpose: a `.plan` is valid only for the TensorRT
version, GPU architecture and driver that built it, so it is rebuilt on each target
from `policy.onnx` (see `deploy/README.md`).

Training config: `experiments/2026-08-19_11-03-32_week04_nohead/params/`, verified
against current code by `scripts/verify_repro.py`.
