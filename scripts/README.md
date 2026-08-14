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
