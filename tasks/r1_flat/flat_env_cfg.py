# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R1 flat-ground velocity-tracking task (Week02 manager-based skeleton).

Adapted from isaaclab_tasks' H1 example
(isaaclab_tasks/manager_based/locomotion/velocity/config/h1/). Differences
from that template, per the Week02 plan:

- Flat ground only, no rough terrain / height scanner -- not in this
  project's scope (the 12-week plan never trains on rough terrain).
- Observations are split into a ``policy`` group (pure proprioception --
  everything actually available on the deployed R1) and a ``critic`` group
  (adds privileged sim-only state). rsl_rl's vec-env wrapper auto-detects an
  observation group literally named "critic" and feeds it to the value
  function only -- this *is* the asymmetric actor-critic split Week03's
  training will use; nothing else needs to change to turn it on.
- Termination is height/tilt-based (``root_height_below_minimum`` +
  ``bad_orientation``) rather than torso-contact-based like H1, since R1's
  link names differ from H1's and the plan asks for "base height too low or
  tilt too large" specifically.
- Rewards are the generic template defaults with R1's foot link name
  (``*_ankle_roll_link``, see assets/r1/r1.py) swapped in. Real reward
  shaping is Week03's job (FR-T3) -- this week's config just needs to not
  crash when stepped.
"""

import math
import sys
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets" / "r1"))
from r1 import R1_CFG  # noqa: E402

##
# Scene
##


@configclass
class R1SceneCfg(InteractiveSceneCfg):
    """Flat-ground scene: R1 + a contact sensor for foot/fall detection."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = R1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Velocity command specification. Ranges are the generic template defaults --
    Week03 narrows these for first training (plan: start with lin_vel_x in [0, 0.5])."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action = scaled joint position offset from R1's default standing pose (assets/r1/r1.py)."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Two groups, deliberately split now so Week03's asymmetric-AC training needs no rework."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Pure proprioception: only what's actually measurable on the deployed R1.
        No base linear velocity -- that's not observable without external tracking."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Proprioception (unnoised) + privileged sim-only state. rsl_rl feeds this
        group to the value function only -- see module docstring."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        # -- privileged additions (sim-only, not observable on the real robot)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_incoming_wrench = ObsTerm(
            func=mdp.body_incoming_wrench, params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis_link")}
        )
        # NOTE: a ground-friction observation is deliberately not here yet -- there's
        # nothing meaningful to observe until Week04's friction-randomization DR
        # event (FR-T5) actually makes it vary per env.

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Episodic reset only. Domain randomization (friction/mass/push/delay, FR-T5)
    is Week04's job -- deliberately not added yet."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )


@configclass
class RewardsCfg:
    """Generic velocity-tracking rewards with R1's foot link name filled in.
    Placeholder for Week03 (FR-T3) -- just needs to be valid, not tuned."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )


@configclass
class TerminationsCfg:
    """"摔倒": base height too low, or tilted too far. Thresholds are a first
    pass (default standing pelvis height is 0.72m, see assets/r1/r1.py) --
    revisit once real standing/training data exists."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.7})


##
# Environment configuration
##


@configclass
class R1FlatEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the R1 flat-ground velocity-tracking environment."""

    scene: R1SceneCfg = R1SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # decimation=10 at dt=0.002 keeps the same ~50Hz control rate as the
        # H1 template's decimation=4 @ 0.005, but at 500Hz physics instead of
        # 200Hz. Required at R1's standing-check gains (assets/r1/r1.py,
        # hip/knee/ankle stiffness ~1200): 200Hz was numerically unstable
        # for this stiffness regardless of gains, 500Hz holds a stable stand.
        # See scripts/README.md.
        self.decimation = 10
        self.episode_length_s = 20.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class R1FlatEnvCfg_PLAY(R1FlatEnvCfg):
    """Small/deterministic variant for interactive play/verification runs."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
