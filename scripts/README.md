# scripts/

Tools that operate on `assets/r1/` and `tasks/r1_flat/`. All are run through
Isaac Lab's launcher **from the project root** (not from inside `scripts/`):

```bash
~/IsaacLab/isaaclab.sh -p scripts/<name>.py [args]
./scripts/<name>.sh [args]
```

| Script | What it does | Output |
|---|---|---|
| `convert_r1_urdf.py` | Converts `assets/r1/R1.urdf` → `assets/r1/usd/R1.usd` via Isaac Lab's `UrdfConverter`. Run once, or whenever the URDF changes. | `assets/r1/usd/` (gitignored build artifact) |
| `inspect_r1.py` | Spawns R1 from `R1_CFG`, holds the default pose under PD for `--settle-steps` (default 5500 ≈ 11s at dt=0.002), then reports per-joint limits/drive gains and a mass sanity check. Also saves a standing screenshot. `--trace-every N` prints height/roll/pitch periodically (0 to disable) — this is what diagnosed the Week02 standing-collapse issue below. Covers both Week01's asset-verification deliverable and Week02's ≥10s standing check. | `docs/joint_check.md`, `docs/r1_standing.png` |
| `random_agent_r1.py` | Our equivalent of Isaac Lab's own `scripts/environments/random_agent.py` — random actions against `Isaac-Velocity-Flat-R1-v0` for `--steps` steps, to confirm the task registers and steps without crashing. We can't use the official script directly: it only imports `isaaclab_tasks`, so it has no way to know about `tasks/r1_flat` living outside Isaac Lab. Week02's task-skeleton deliverable. | stdout only |
| `train_r1.py` | Our equivalent of Isaac Lab's own `scripts/reinforcement_learning/rsl_rl/train.py` — same reason we can't use the official script directly (it only imports `isaaclab_tasks`). Trains `Isaac-Velocity-Flat-R1-v0` with rsl_rl PPO, asymmetric AC (auto-detected from the env's `critic` observation group — see `tasks/r1_flat/agents/rsl_rl_ppo_cfg.py`). Week03's first-training deliverable (FR-T4). Supports checkpoint-resume (`--resume --load_run <run_id> --checkpoint <name>`) to pause training, inspect the gait with `play_r1.py`, then continue toward the same target — see the module docstring for the exact workflow and the gotcha below. | `logs/rsl_rl/r1_flat/<run_id>/` (gitignored — checkpoints + tensorboard events) |
| `play_r1.py` | Our equivalent of Isaac Lab's own `scripts/reinforcement_learning/rsl_rl/play.py` (same external-task-package reason). Loads a checkpoint and steps `Isaac-Velocity-Flat-R1-Play-v0` with it for visual/recorded verification. Headless has no GUI, so pass `--video` or it spins forever. Week03 step 4 (deferred as of the first training run — see `experiments/`). | `logs/rsl_rl/r1_flat/<run_id>/videos/play/` when `--video` is passed; also exports `policy.pt`/`policy.onnx` next to the checkpoint |
| `diagnose_gait.py` | Rolls out a checkpoint (no rendering) and measures **per-foot** gait statistics: swing count, mean swing duration, **air-time fraction**, plus each foot's dominant stepping frequency and the left/right phase offset. Written after three reward-shaping rounds were misjudged from tensorboard values and sampled video frames — air-time fraction (~0.4-0.5 per foot for a real walk) is the number that actually distinguishes walking from a one-legged hop. Use this as the pass/fail check after any gait-related training run. | stdout + `logs/rsl_rl/r1_flat/<run_id>/gait_diagnosis.txt` |
| `eval_baseline.py` | Rolls out a checkpoint at a grid of *fixed* commanded forward speeds and reports per speed: linear-velocity tracking error (**the PG-1 acceptance number**), falls, energy, action smoothness. Pins commands by collapsing the command term's ranges to a point value rather than overwriting `vel_command_b` each step, so the term's own resampling can't fight it. `--friction` / `--push_vel` turn it into a stress test, which is how the with-DR vs without-DR comparison is produced. Complementary to `diagnose_gait.py`: this says whether the robot *goes the commanded speed*, that one says whether it *walks* while doing so. | stdout + `logs/rsl_rl/r1_flat/<run_id>/baseline[_tag].md` |
| `archive_run.py [run_id]` | Copies a training run's config + final tensorboard metrics into `experiments/` for ablation comparisons across weeks. Doesn't need Isaac Sim — run with plain `python`, not `isaaclab.sh`. See `experiments/README.md`. | `experiments/<run_id>/`, `experiments/runs.md` |
| `verify_repro.py --run <run_id>` | Rebuilds the env/agent config from current code and diffs it field by field against a run's archived `params/*.yaml`, turning that dump from a record into a contract the repo is tested against (FR-T6). Exits non-zero on drift. Values the scene resolves at construction (`{ENV_REGEX_NS}`, terrain env count/spacing) are normalised rather than ignored, so every reported difference is real; invocation-dependent fields are listed separately. Sanity-check it against a superseded run — `week04_hwspec` reports exactly the two fields the head removal changed. | stdout + `experiments/<run_id>/repro_check.txt` |
| `plot_m1_figures.py` | Renders the M1 gate-review figures from data already on disk — the baseline reports `eval_baseline.py` wrote and the tensorboard scalars the runs logged — so the figures can't drift from the numbers they claim to show. Plain `python`, no Isaac Sim. Stitches `week04_dr`'s two event files (it was trained then resumed) and drops the first 25 post-resume iterations, where rsl_rl's reward buffer is refilling and reports a dip that isn't real. | `docs/m1_dr_robustness.png`, `docs/m1_training_curves.png` |
| `throughput_sweep.sh` | Sweeps `num_envs` on the official `Isaac-Velocity-Flat-H1-v0` task from 64 up to 16384, recording steady-state fps and peak GPU memory per level. This is Week01's RK-7 throughput baseline — run *before* touching the R1 task, not on it. | `docs/throughput_sweep.md` |
| `throughput_sweep_extend.sh <N> [<N> ...]` | Continues the sweep at specific `num_envs` levels without re-running the ones already done — used to push up to the actual OOM ceiling. Appends to the same report. | `docs/throughput_sweep.md` |

## Known gotchas (already worked around in the code, documented here so nobody re-discovers them the hard way)

- **Don't `set -u` in a script that calls `./isaaclab.sh`.** Bash exports shell
  options like `nounset` to child scripts via `SHELLOPTS`, and IsaacLab's own
  `setup_conda_env.sh` references `$ZSH_VERSION` unguarded — it aborts instantly
  under nounset. `throughput_sweep*.sh` use `set -o pipefail` only.
- **`inspect_r1.py` (and `convert_r1_urdf.py`'s own post-conversion introspection)
  can hang after writing their output.** They sometimes don't cleanly exit the
  Isaac Sim process even after finishing all real work. Check that the expected
  output file's mtime updated; if the process is still alive afterward with no
  GPU activity, it's safe to `kill` it — the output was already written. Always
  check `ps aux | grep isaac` before launching another run — these pile up fast
  and multiple simultaneous Isaac Sim instances will corrupt each other's runs
  (contend for GPU, or crash one outright).
- Camera pose (`camera.set_world_poses_from_view`) must be set **after**
  `sim.reset()`, not before — the camera isn't initialized until then.
- **rsl_rl's `--max_iterations` means different things depending on `--resume`.**
  On a fresh run it's the total iteration count. On a `--resume` run, upstream
  rsl_rl/IsaacLab treat it as *additional* iterations on top of wherever the
  loaded checkpoint left off (`OnPolicyRunner.learn` does
  `tot_iter = current_learning_iteration + num_learning_iterations`) — passing
  the same value you used originally silently trains way past your intended
  stopping point. `train_r1.py` re-interprets `--max_iterations` as the
  *absolute* target when `--resume` is set (computes the remaining count
  itself) specifically to support pause-inspect-resume workflows; verified
  with a small-scale test (trained to it=4, resumed with `--max_iterations 8`,
  correctly ran 4 more iterations to reach it=7/absolute-target-8, not
  it=4+8=12). Keep this in mind if you ever call `OnPolicyRunner` directly or
  compare against upstream IsaacLab docs/scripts — their semantics differ from
  ours on purpose.
- **Isaac Sim's shutdown can eat a script's stdout.** `simulation_app.close()`
  can kill the process before Python flushes a block-buffered stdout, so a
  report printed at the end of a run vanishes when you redirect to a file
  (it survives on a terminal, where stdout is line-buffered — so this looks
  intermittent). `diagnose_gait.py` builds its report as one string and
  writes it with `flush=True` plus a copy to disk. For other scripts, run
  them with `PYTHONUNBUFFERED=1`.
- **`feet_air_time_positive_biped` is maximized by standing on one leg.**
  With one foot permanently planted and the other permanently in the air,
  `single_stance` is always true and both feet's `in_mode_time` grow without
  bound, so the term sits clamped at its `threshold` maximum forever — never
  taking a step is its global optimum. Three Week03 reward rounds were spent
  patching around this before it was found (`experiments/*/NOTES.md`). If you
  use this term, pair it with a touchdown gate (`compute_first_contact`) or a
  max-air-time penalty, and **never read a rising `feet_air_time` as evidence
  of a better gait** — check `diagnose_gait.py`'s air-time fractions instead.
- **High joint-PD stiffness needs a correspondingly fine physics timestep.**
  R1's standing gains (`assets/r1/r1.py`, hip/knee/ankle stiffness ~1200
  N·m/rad) fell over identically regardless of *any* gain combination at the
  default 1/60s timestep — including gains well past the stiffness a
  whole-body inverted-pendulum analysis said should be sufficient. It was a
  numerical instability in PhysX's implicit joint-drive solver, not a physical
  or tuning problem: switching to `dt=0.002` (500Hz) with the *same* gains
  fixed it completely (rock-stable for 10+ s). If a stiff articulation is
  falling over in a way that's insensitive to gain changes, suspect the
  timestep before the controller. `tasks/r1_flat/flat_env_cfg.py` uses the
  same `dt=0.002` (with `decimation=10` to keep the ~50Hz control rate).
- **Check the robot's own spec before inventing actuator numbers.** Week02 set
  R1's leg effort limits to 150 N·m, chosen to pass a "hold the pose under pure
  joint PD for 10s" standing test. `R1.urdf` declares 60 (hip/knee) and 50
  (ankle), and Unitree's official RL config agrees exactly — so the sim robot
  was 2.5-3x stronger than the hardware, and the Week04 policy spent 11.6% of
  its time commanding ankle torques no real R1 could produce. Two lessons:
  (1) the passive-standing criterion is wrong for an RL task — the policy
  re-targets every joint at 50Hz and balances *actively*, so it never needs the
  pose to be passively self-supporting; (2) **no in-sim metric can catch this.**
  PG-1 passed, falls were zero, the gait diagnosis was clean. It took comparing
  against an external reference. After the correction (100/2/60 legs, 40/2/50
  ankles, per-group action scale `0.25*effort/stiffness`), tracking error,
  robustness, energy and gait symmetry all improved *simultaneously*.
- **R1's leg gains only stand up under an *implicit* actuator.** The corollary
  of the entry above, found while adding Week04's control-delay randomization:
  swapping the legs onto `DelayedPDActuatorCfg` (an *explicit* actuator —
  PD computed in Python, applied as an effort) made R1 collapse in 1.5s, and it
  collapsed identically with the delay set to zero, so the actuator model was
  at fault rather than the lag. Explicit damping needs roughly `dt < 2J/d`, and
  the ankles run `d=150` against an inertia of order 0.01 kg·m² (armature
  included) — a limit near 1e-4 s against the 2e-3 s this env uses. The failure
  order matched: roll broke first (-10.6° at t=0.5s) while pitch was still
  1.3°, i.e. the ankles went before anything else. Control delay is modeled at
  the **action** level instead (`DelayedJointPositionAction` in
  `tasks/r1_flat/mdp.py`), which is both safe and the more faithful unit — a
  policy-loop lag in 20ms control steps, not 2ms physics substeps.
- **An observation term's function is called once before startup events run.**
  `ObservationManager` invokes each term while preparing it, to discover its
  output shape, and that happens *before* `load_managers` applies `mode="startup"`
  events. So a term that caches an expensive lookup on first use will cache the
  **pre-randomization** value and then return a constant forever. This bit the
  critic's ground-friction observation in Week04: all envs read exactly 1.000
  (the USD default) while the actual friction spanned 0.6-1.2. Fix is to drop
  the cache once on the first `reset()`, which happens after startup. More
  generally: after wiring up any domain randomization, **measure that it varies
  across envs** before spending a training run on it.
