# scripts/

Command-line tools for the simulation side: building and inspecting the robot model,
training and evaluating the policy, the simulated half of the robustness sweep, the
figures, and a few repository checks.

## How to run them

From the **repository root**, not from inside `scripts/`. Scripts that start Isaac Sim go
through Isaac Lab's launcher; the others are plain Python:

```bash
~/IsaacLab/isaaclab.sh -p scripts/<name>.py --headless [args]   # Isaac Sim scripts
python3 scripts/<name>.py [args]                                # "plain python" below
./scripts/<name>.sh [args]
```

Every script documents its arguments in its module docstring (`--help` works too).

## The scripts

| script | what it is for | writes |
|---|---|---|
| **robot model** | | |
| `convert_r1_urdf.py` | build the USD from `assets/r1/R1.urdf`; run once after cloning | `assets/r1/usd/` (not tracked) |
| `inspect_r1.py` | joint limits, gains and masses as simulated, and a passive standing check | `docs/joint_check.md` |
| `random_agent_r1.py` | smoke test: the task registers and steps with random actions | stdout |
| **training and evaluation** | | |
| `train_r1.py` | train the policy with rsl_rl PPO | `logs/rsl_rl/r1_flat/<run_id>/` (not tracked) |
| `play_r1.py` | replay a checkpoint, record a video, export TorchScript and ONNX | `<run>/videos/`, `<run>/exported/` |
| `eval_baseline.py` | speed-tracking error, falls and energy at fixed commanded speeds (the PG-1 number) | `<run>/baseline[_tag].md` |
| `diagnose_gait.py` | per-foot gait statistics: does it actually walk? | `<run>/gait_diagnosis.txt` |
| `archive_run.py` | copy a run's config and final metrics into `experiments/` (plain python) | `experiments/<run_id>/` |
| `verify_repro.py` | check that current code still produces a run's exact config | `experiments/<run_id>/repro_check.txt` |
| **robustness sweep and figures** | | |
| `sweep_gain_robustness.py` | simulated half of the `kp_scale` sweep | `outputs/gain_sweep/<date>/` |
| `plot_gain_sweep.py` | the stage A figure: sim curve and real points on one axis (plain python) | `docs/stageA_kp_sweep.png` |
| `plot_m1_figures.py` | the training-milestone figures (plain python) | `docs/m1_*.png` |
| `record_demo_sim.py` | film the robot walking a `mission_ctl` script in simulation | an `.mp4` |
| `make_demo_video.py` | assemble the demo video (plain python + ffmpeg) | an `.mp4` |
| **checks** | | |
| `check_doc_links.py` | every relative link and anchor in the Markdown resolves (plain python; in `run_tests.sh`) | stdout |
| `throughput_sweep.sh`, `throughput_sweep_extend.sh` | how many parallel envs this GPU can train with, and how fast | `docs/throughput_sweep.md` |

## Details

### Why there are local copies of Isaac Lab's scripts

`random_agent_r1.py`, `train_r1.py` and `play_r1.py` mirror Isaac Lab's own
`random_agent.py`, `train.py` and `play.py`. The originals only `import isaaclab_tasks`,
so they cannot see a task package that lives outside Isaac Lab, such as `tasks/r1_flat/`.
The copies register it first and otherwise behave the same.

### `train_r1.py`

- Asymmetric actor-critic is picked up automatically from the task's `critic`
  observation group (`tasks/r1_flat/agents/rsl_rl_ppo_cfg.py`).
- `--resume --load_run <run_id> --checkpoint <name>` continues a run, so you can stop,
  inspect the gait with `play_r1.py`, and carry on.
- `--max_iterations` is the **absolute** target even with `--resume`; upstream treats it
  as additional iterations (see *Known pitfalls*).

### `play_r1.py`

Headless mode has no window, so pass `--video`, or it runs forever. It also exports
`policy.pt` and `policy.onnx` next to the checkpoint; that is where `models/` came from.

### `eval_baseline.py` and `diagnose_gait.py`

The two answer different questions and both are needed after a training run.

- `eval_baseline.py`: **does the robot go the commanded speed?** It pins each command by
  collapsing the command range to a point, so the command term's own resampling cannot
  fight it. `--friction` and `--push_vel` turn it into a stress test.
- `diagnose_gait.py`: **does it walk while doing so?** It reports per-foot swing count,
  swing duration, air-time fraction (about 0.4–0.5 per foot for a real walk), each
  foot's stepping frequency and the left/right phase offset. It was written after
  reward changes were misjudged three times from tensorboard curves and video frames.

### `verify_repro.py`

Rebuilds the environment and agent config from the current code and diffs it field by
field against a run's archived `params/*.yaml`, turning that dump into a contract. It
exits non-zero on drift. Values the scene resolves at construction time are normalized
rather than ignored, and fields that vary per invocation (env count, device) are listed
separately. Try it on a superseded run: `week04_hwspec` reports exactly the two fields
the head removal changed.

### `inspect_r1.py`

Holds the default pose under joint PD at the training task's physics step (`--dt`,
default 0.005 s) for `--settle-seconds` (default 11), then writes the joint table.
`--trace-every N` prints height and tilt as it goes. With the hardware gains the robot
does **not** stand passively. That is expected: the policy balances it actively. The
standing screenshot `docs/r1_standing.png` comes from W02's much stiffer gains and is
only overwritten when the robot stands.

### `sweep_gain_robustness.py` and `plot_gain_sweep.py`

The sweep runs every gain in one rollout, with per-env actuator stiffness. Each env drives
its own `mission_ctl` executor with the exact command string the robot walked, and is
scored over the commanded window with the same stability definition as the robot. It
refuses to run unless the policy reproduces the deployed parity fixture.
`plot_gain_sweep.py` imports the robot-side reduction from
`deploy/tools/probe_cpp/gain_sweep_real.py`, so the figure cannot disagree with it.

### `plot_m1_figures.py`

Builds the figures from data already on disk: the `eval_baseline.py` reports and the
runs' tensorboard scalars. Figures cannot drift from the numbers. It stitches
`week04_dr`'s two event files (trained, then resumed) and drops the first 25
post-resume iterations, where rsl_rl's reward buffer refills and shows a dip that is not
real.

### `throughput_sweep*.sh`

Run on Isaac Lab's own `Isaac-Velocity-Flat-H1-v0` task, before the R1 task existed:
`num_envs` from 64 to 16384, steady-state fps and peak GPU memory per level;
`throughput_sweep_extend.sh <N>...` adds levels up to the out-of-memory point.

## Known pitfalls

All are already worked around in the code; they are written down so nobody rediscovers them the hard way.

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
  taking a step is its global optimum. Three W03 reward rounds were spent
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
  timestep before the controller. The task used `dt=0.002` / `decimation=10`
  through W04's first run; once the gains were corrected to the hardware's
  100/40 (next entry) the constraint went away, and the deployed policy trained
  at `dt=0.005` / `decimation=4` (`tasks/r1_flat/flat_env_cfg.py`).
- **Check the robot's own spec before inventing actuator numbers.** W02 set
  R1's leg effort limits to 150 N·m, chosen to pass a "hold the pose under pure
  joint PD for 10s" standing test. `R1.urdf` declares 60 (hip/knee) and 50
  (ankle), and Unitree's official RL config agrees exactly — so the sim robot
  was 2.5-3x stronger than the hardware, and the W04 policy spent 11.6% of
  its time commanding ankle torques no real R1 could produce. Two lessons:
  (1) the passive-standing criterion is wrong for an RL task — the policy
  re-targets every joint at 50Hz and balances *actively*, so it never needs the
  pose to be passively self-supporting; (2) **no in-sim metric can catch this.**
  The speed-tracking gate (PG-1) passed, falls were zero, the gait diagnosis was clean. It took comparing
  against an external reference. After the correction (100/2/60 legs, 40/2/50
  ankles, per-group action scale `0.25*effort/stiffness`), tracking error,
  robustness, energy and gait symmetry all improved *simultaneously*.
- **R1's leg gains only stand up under an *implicit* actuator.** The corollary
  of the entry above, found while adding W04's control-delay randomization:
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
  critic's ground-friction observation in W04: all envs read exactly 1.000
  (the USD default) while the actual friction spanned 0.6-1.2. Fix is to drop
  the cache once on the first `reset()`, which happens after startup. More
  generally: after wiring up any domain randomization, **measure that it varies
  across envs** before spending a training run on it.

## History

- W01: `convert_r1_urdf.py`, `inspect_r1.py`, the throughput sweep (risk item RK-7: is
  one RTX 2080 Ti fast enough to train on?).
- W02: `random_agent_r1.py` (the task skeleton); `inspect_r1.py`'s standing check.
- W03: `train_r1.py`, `play_r1.py`, `diagnose_gait.py`.
- W04: `eval_baseline.py`, `archive_run.py`, `verify_repro.py`, `plot_m1_figures.py`.
- Stage A: `sweep_gain_robustness.py`, `plot_gain_sweep.py`, the demo scripts.
- W01–W08 and stages A–D are explained in [docs/project_history.md](../docs/project_history.md).
