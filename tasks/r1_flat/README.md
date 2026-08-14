# tasks/r1_flat/

`Isaac-Velocity-Flat-R1-v0` — R1's flat-ground velocity-tracking task
(Week02: manager-based task skeleton + standing check).

| File | What's in it |
|---|---|
| `__init__.py` | `gym.register()` for `Isaac-Velocity-Flat-R1-v0` and its `-Play-v0` variant. No `rsl_rl_cfg_entry_point` yet — that's added when Week03 starts training. |
| `flat_env_cfg.py` | `R1FlatEnvCfg`: scene, actions, observations, events, rewards, terminations. See the module docstring for what's adapted from Isaac Lab's H1 example vs. R1-specific. |

## What's real vs. placeholder right now

- **Actions, scene, task registration**: real, verified (`scripts/random_agent_r1.py` steps it without crashing).
- **Observations**: real design — `policy` group is pure proprioception (deployable), `critic` group adds privileged state (base linear velocity, incoming wrench). rsl_rl auto-detects the `critic` group name for asymmetric actor-critic training; this is set up now so Week03 needs no rework.
- **Rewards**: generic template placeholders with R1's foot link name filled in. Not tuned — that's Week03 (FR-T3).
- **Events (domain randomization)**: episodic reset only. Friction/mass/push/delay DR is Week04 (FR-T5) — deliberately not here yet.

## The standing-gains / timestep dependency

`R1_CFG` in `assets/r1/r1.py` uses high leg-joint PD stiffness (~1200 N·m/rad)
to hold R1's default stance (see that file's comments for the full diagnosis —
short version: a whole-body inverted-pendulum problem, not per-joint
compliance, and it also required a finer physics timestep to be numerically
stable). This env's `self.sim.dt = 0.002` / `self.decimation = 10` matches
that requirement. If either the gains or the timestep change, re-check
standing stability with `scripts/inspect_r1.py` before assuming training will
behave sanely — a numerically unstable stiff articulation will look like a
policy that can't learn to stand, not like a config bug.
