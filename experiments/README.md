# experiments/

Git-tracked training-run records, for comparing runs across ablations
(W03 first training, W04 domain randomisation) without needing to keep every
raw artifact around. W03/W04 are weeks of the project plan; see *How the project
was organised* in the [top-level README](../README.md#how-the-project-was-organised).

## The runs, in order

Each folder's `NOTES.md` is the original lab note, in Chinese. In one line each:

| run | changed | outcome |
|---|---|---|
| `2026-08-14_14-22-09_first_train` | W03 baseline: H1 template rewards on R1 | no falls, but a wide-legged shuffle; the air-time reward never produced a gradient |
| `2026-08-14_16-19-36_reward_fix` | heavier air-time and foot-slide terms, hip deviation split, base-height term | real steps, but the right foot taps to collect the air-time reward |
| `2026-08-15_10-56-44_reward_fix_1500` | same config, 1500 iterations instead of 800 | asymmetry unchanged: a flaw in the reward, not training length |
| `2026-08-15_13-25-00_reward_fix_symmetry` | left/right symmetry penalty on the *instantaneous* duty cycle | **failed**: penalised every single-stance moment, so the policy hopped with both feet |
| `2026-08-15_15-37-12_reward_fix_symmetry_v2` | symmetry penalty on *completed* swings | **failed**, first misjudged as a success by eye; `scripts/diagnose_gait.py` was written here and showed one-leg support |
| `2026-08-15_17-34-57_touchdown_gate` | `feet_air_time_positive_biped` removed (standing on one leg maximises it); reward paid at touchdown for a 0.15–0.45 s swing; penalty for a foot held up | **the first alternating gait** |
| `2026-08-17_17-38-37_week04_dr` | 5-frame observation history, five-item domain randomisation, commands to 1.0 m/s and turning, curriculum | PG-1 passed (worst 0.084 m/s), but ankle torques beyond the real robot's rating |
| `2026-08-18_15-33-21_week04_hwspec` | actuator limits and gains corrected to the URDF / Unitree spec | every metric improved at once (worst 0.050 m/s); new defect: the head pitched up to its joint limit |
| `2026-08-19_11-03-32_week04_nohead` | head removed from the action space: **24-dim action, 425-dim observation** | **the deployed policy**: worst 0.037 m/s, 0.076 under friction 0.6 + pushes; copied to [`models/week04_nohead/`](../models/week04_nohead/) |

PG-1 is the training gate: forward-speed tracking error ≤ 0.15 m/s over 0.5–1.0 m/s in
simulation.

## What is kept here

Stage A's `kp_scale` robustness sweep needs **no retraining** -- it
re-evaluates the deployed checkpoint at different actuator gains -- so it adds
no rows here; its results are in [`docs/stageA_kp_sweep.md`](../docs/stageA_kp_sweep.md).
(The originally planned quantisation × robustness comparison was dropped with
INT8 on 2026-09-26.)

This is deliberately separate from `logs/` (git-ignored): `logs/rsl_rl/<task>/<run_id>/`
holds the *raw* rsl_rl output -- model checkpoints (`*.pt`), full tensorboard
event binaries, per-iteration everything. That's regenerable from a config
and too big for git. `experiments/` holds only what's needed to answer "what
config produced what result" later, without re-running anything:

| Path | Contents |
|---|---|
| `runs.md` | One-row-per-run index: date, task, key hyperparams, final metrics, git commit. Skim this first when comparing runs. |
| `<run_id>/params/env.yaml`, `<run_id>/params/agent.yaml` | Full dumped env + agent config for that run (copied from `logs/rsl_rl/<task>/<run_id>/params/`) -- the exact ground truth, `diff`-able between two runs to see what an ablation actually changed. |
| `<run_id>/summary.md` | Final metrics (mean reward, episode length, termination breakdown, reward-term breakdown) pulled from that run's tensorboard log, plus the git commit/dirty-state the run was trained against. |
| `<run_id>/NOTES.md` | Hand-written lab note (Chinese): the problem, what changed from the previous run, the result and the diagnosis. Summarised in English in the table above. |
| `<run_id>/pip_freeze.txt`, `<run_id>/repro_check.txt` | W04 runs only: the Python environment, and `scripts/verify_repro.py`'s check that current code still produces this config. |

## Usage

After a training run (running or finished -- it reads whatever's been
logged so far), archive it:

```bash
conda activate env_isaaclab   # no Isaac Sim needed for this step, plain python is enough
python scripts/archive_run.py                      # archives the most recent run under logs/rsl_rl/r1_flat/
python scripts/archive_run.py 2026-08-14_14-22-09   # or a specific run_id
```

This copies the two config yamls, extracts final tensorboard metrics into
`summary.md`, and appends a row to `runs.md`. Re-running it against the same
`run_id` overwrites that run's entry (useful to re-archive once a run that
was still training when first archived has finished).
