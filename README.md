# R1process

Working directory for the R1 Sim2Real project's Isaac Lab side: R1 asset pipeline,
the R1 manager-based task, per-week verification scripts, and their output
reports. This is the practical counterpart to the
[12-week dev plan](~/kdw/development%20plan/) — currently at **Week04 in progress**
(Week03 produced R1's first real alternating gait; see `experiments/`).

## Layout

- `assets/` — the R1 robot definition: URDF, meshes, Isaac Lab `ArticulationCfg`,
  and the USD build artifact. See `assets/README.md`.
- `tasks/` — the R1 flat-ground velocity-tracking gym task (manager-based:
  scene/actions/observations/rewards/terminations/events). See `tasks/README.md`.
- `scripts/` — tools that operate on the asset and task: URDF→USD conversion,
  load/limit/standing verification, task smoke test, GPU throughput sweeps.
  See `scripts/README.md`.
- `docs/` — dated verification reports and screenshots produced by the scripts
  above (Week01/Week02 exit-criteria evidence). See `docs/README.md`.
- `experiments/` — git-tracked training-run records (config + final metrics)
  for comparing runs across ablations. See `experiments/README.md`.

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

R1 is the **EDU version** with an onboard 8-core Jetson Orin NX (40-100 TOPS) —
no external compute board needed for on-robot deployment (Week05+).
