# experiments/

Git-tracked training-run records, for comparing runs across ablations
(Week03 first training, Week04 DR ablation, Week10 quantization x
robustness, etc.) without needing to keep every raw artifact around.

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
