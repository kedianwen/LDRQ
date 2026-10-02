# tasks/r1_flat/

`Isaac-Velocity-Flat-R1-v0`: R1 walking on flat ground, tracking a commanded forward
speed and yaw rate. The deployed policy
([`models/week04_nohead/`](../../models/week04_nohead/)) was trained on this task as it
stands.

| file | what's in it |
|---|---|
| `__init__.py` | `gym.register()` for `Isaac-Velocity-Flat-R1-v0` and its `-Play-v0` variant (fewer envs, no randomization), both with the PPO config as `rsl_rl_cfg_entry_point` |
| `flat_env_cfg.py` | `R1FlatEnvCfg`: scene, commands, actions, observations, randomization events, rewards, terminations, command curriculum. Each config class's docstring says why it is the way it is |
| `mdp.py` | R1-specific terms not in Isaac Lab: the touchdown-gated swing reward, the command-range curriculum, the action-level control delay |
| `symmetry.py` | the left/right mirror map for PPO's symmetry augmentation |
| `agents/rsl_rl_ppo_cfg.py` | PPO hyperparameters, seed 42, 3000 iterations, symmetry augmentation on |

## The task in one table

| | |
|---|---|
| **action** | 24 joint-position targets (legs 2×6, waist 2, arms 2×5; the head is not actuated), per-group scale `0.25 × effort / stiffness`, at 50 Hz (`sim.dt = 0.005`, `decimation = 4`) |
| **policy observation** | 85 floats per frame, proprioception only (angular velocity, gravity direction, command, joint positions and velocities, last action), the last 5 frames stacked → **425**. No base linear velocity: the robot cannot measure it |
| **critic observation** | adds privileged simulator state (base linear velocity, external wrench, foot friction): asymmetric actor-critic |
| **commands** | forward 0 → 1.0 m/s, yaw rate ±0.5 rad/s, reached by a curriculum over the first 1000 iterations; **sideways is pinned to 0 and there is no backward**, which is why the robot refuses both |
| **randomization** | friction 0.6–1.2, body mass ±10 %, actuator gains ×0.85–1.15, pushes ±0.5 m/s every 10–15 s, control delay 0–1 step |
| **rewards** | velocity tracking; a swing reward paid only at touchdown for a 0.15–0.45 s swing; penalties including feet held up, foot slide, joint limits, upper-body and hip drift, falling |
| **termination** | base below 0.51 m, tilt beyond 0.7 rad, or the 20 s episode ends |

## Traps this task already walked into

Each is fixed in the code and explained where it is fixed; listed so nobody undoes one:

- **`feet_air_time_positive_biped` is maximized by standing on one leg.** It is not used.
  The swing reward is paid at touchdown instead (`mdp.py`). W03 spent three training runs
  on it; see [`experiments/README.md`](../../experiments/README.md).
- **Any actuated joint without a reward on it becomes a balance aid.** The head ended up
  pinned at its limit, so it was taken out of the action space (24 actions, not 26).
- **Actuator limits come from the robot's spec, not from what makes standing pass.**
  The W02 values were 2.5–3× stronger than the hardware.
- **Stiff leg gains need a fine timestep and an implicit actuator.** With the W02 gains
  (~1200 N·m/rad) the robot collapsed at any coarser step than `dt = 0.002`, whatever the
  gains; with an explicit PD actuator it collapsed even then. Since the gains were
  corrected to the hardware's 100/40 (W04), `dt = 0.005` is stable and is what the
  deployed policy trained at. If you raise the gains, re-check standing with
  `scripts/inspect_r1.py` before training.

Longer write-ups of each are in [`scripts/README.md`](../../scripts/README.md#known-pitfalls).

## History

W02: the task skeleton and standing; W03: rewards and the first real gait; W04: domain
randomization, the observation history and the final 24-action / 425-observation
interface. Weeks are explained in [docs/project_history.md](../../docs/project_history.md).
