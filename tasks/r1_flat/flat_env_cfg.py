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
- Rewards (Week03, FR-T3): tracking terms + physical-plausibility penalties
  (tilt, torque, joint limits, foot slide) + upper-body joint_deviation
  penalties (R1 has 14 non-leg DOF with no task of their own yet -- without
  a "stay near default" penalty the policy can exploit arm-flailing for
  balance, which is free in sim but doesn't transfer and fights Week0X's
  future arm-task training). See RewardsCfg docstring for the full rationale.
- Commands are narrowed for first training: lin_vel_x in [0, 0.5], lin_vel_y
  and ang_vel_z pinned to 0 -- straight-line walking only. Widen once this
  is learned (plan explicitly time-boxes this: "只求学起来，不求达标").
"""

import math
import sys
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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

from tasks.r1_flat.mdp import (
    DelayedJointPositionActionCfg,
    body_material_friction,
    command_range_curriculum,
    feet_air_time_excess_l1,
    feet_swing_touchdown,
)

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


# Week04 (FR-T4): frames of proprioception stacked into the actor's observation.
# 5 frames at the 50Hz control rate = 100ms of history, enough to span most of a
# 1.34Hz gait cycle's swing phase (the Week03 policy's measured step frequency).
OBS_HISTORY_LENGTH = 5

# Week04 command curriculum end points. Module-level so the curriculum term, the
# CommandsCfg defaults and the PLAY variant can't drift apart.
COMMAND_RANGES_INITIAL = {"lin_vel_x": (0.0, 0.5), "ang_vel_z": (0.0, 0.0)}
COMMAND_RANGES_FINAL = {"lin_vel_x": (0.0, 1.0), "ang_vel_z": (-0.5, 0.5)}


@configclass
class CommandsCfg:
    """Velocity command specification.

    Week03 first-training range was straight-line walking only (lin_vel_x in
    [0, 0.5], lin_vel_y and ang_vel_z pinned to 0). heading_command is off
    because with heading on, the commanded ang_vel_z is computed from heading
    error and clamped into ranges.ang_vel_z at runtime -- leaving that range
    at (0, 0) would silently fight the heading controller instead of just
    not commanding turns.

    Week04 must widen this: PG-1 is stated over 0.5-1.0 m/s, and measuring the
    Week03 policy (scripts/eval_baseline.py) showed exactly the expected
    shape -- 0.022 m/s tracking error at 0.5 m/s but 0.236 m/s at 1.0 m/s,
    with zero falls. It isn't unstable up there, it simply never trained
    there. Turning is enabled too: PG-1 doesn't ask for it, but Week06's real
    deployment does, and adding it later means retraining.

    The ranges here are the curriculum's *starting* values (Week03's, which
    already work); CurriculumCfg.command_ranges ramps them to the final
    lin_vel_x (0, 1.0) / ang_vel_z (-0.5, 0.5). lin_vel_y stays pinned at 0 --
    humanoid lateral stepping is much harder than turning and is rarely
    commanded on hardware, not worth the risk in an M1 gate week.
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=COMMAND_RANGES_INITIAL["lin_vel_x"],
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=COMMAND_RANGES_INITIAL["ang_vel_z"],
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action = scaled joint position offset from R1's default standing pose (assets/r1/r1.py).

    Week04 (FR-T5) adds the fifth domain randomization item here rather than in
    EventCfg: the setpoint reaching the joints lags 0-1 control steps (0-20ms),
    resampled per episode. See DelayedJointPositionAction in tasks/r1_flat/mdp.py
    for why this is not Isaac Lab's DelayedPDActuatorCfg -- short version, that
    one is an explicit actuator and R1's leg gains fall over without PhysX's
    implicit joint-drive solver (measured, not assumed).
    """

    joint_pos = DelayedJointPositionActionCfg(
        asset_name="robot",
        # 24 of R1's 26 joints -- the head is deliberately *not* actuated by the
        # policy, matching Unitree's own R1 config (their MJCF has no head
        # actuator either). It stays PD-held at its default angle by the "head"
        # actuator group in assets/r1/r1.py.
        #
        # It used to be in here, via ".*", and with no reward term mentioning
        # head_pitch/head_yaw the policy drove head_pitch to its *hard* limit
        # (-0.628 rad = -36 deg, 100% of samples, std 0.001) and left it there:
        # R1 walked staring at the sky. The fix is structural rather than a
        # "stay near default" penalty because the head had already become
        # load-bearing -- zeroing its action at inference cost 4-7x tracking
        # error and produced falls at 1.0 m/s. A joint the policy cannot drive
        # cannot be co-opted for balance in the first place. Full comparison of
        # both candidate fixes:
        # ~/kdw/experiment_record/Week04_头部仰天_根因与两方案对比.md
        joint_names=[
            ".*_hip_.*_joint",
            ".*_knee_joint",
            ".*_ankle_.*_joint",
            "waist_.*_joint",
            ".*_shoulder_.*_joint",
            ".*_elbow_joint",
            ".*_wrist_roll_joint",
        ],
        # Per-group scale, following Unitree's own R1 config, which derives it as
        # 0.25 * effort_limit / stiffness -- i.e. a full-scale action asks for a
        # quarter of the joint's torque budget. A single global 0.5 (what this
        # used through Week04) is not scale-free: against the old stiffness of
        # 1200 it meant a 0.05 rad action already commanded 60 N*m, so the
        # controller effectively ran saturated. That is the direct cause of the
        # measured ankle p99 pinned at the torque ceiling.
        scale={
            ".*_hip_.*_joint": 0.15,
            ".*_knee_joint": 0.15,
            ".*_ankle_.*_joint": 0.3125,
            "waist_.*_joint": 0.15,
            ".*_shoulder_pitch_joint": 0.375,
            ".*_shoulder_roll_joint": 0.375,
            ".*_shoulder_yaw_joint": 0.4125,
            ".*_elbow_joint": 0.4125,
            ".*_wrist_roll_joint": 0.4125,
        },
        use_default_offset=True,
        min_delay=0,
        max_delay=1,
    )


@configclass
class ObservationsCfg:
    """Two groups, deliberately split now so Week03's asymmetric-AC training needs no rework."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Pure proprioception: only what's actually measurable on the deployed R1.
        No base linear velocity -- that's not observable without external tracking.

        Week04 (FR-T4) stacks the last OBS_HISTORY_LENGTH frames of every term
        here, which is what lets a pure-proprioception actor *infer* the things
        it can't measure (base linear velocity, ground contact, actuator lag)
        from how its own state evolved. That's this project's replacement for a
        teacher-student distillation stage -- the critic keeps its privileged
        observations, the actor gets memory instead."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            # group-level setting: applies to every term above, 85 -> 425 dims.
            # NOTE the buffer lives in the *env* (ObservationManager's
            # CircularBuffer), not in the network, so Week05's exported ONNX
            # takes a flat 425-dim input and the C++ node on the robot has to
            # maintain this 5-frame queue itself.
            self.history_length = OBS_HISTORY_LENGTH
            self.flatten_history_dim = True

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
        # Week04: friction now *does* vary per env (EventCfg.physics_material),
        # so this became worth observing -- an unobservable friction is reward
        # noise the value function would otherwise have to absorb.
        # asset_cfg must be passed through params, not left to the function's
        # default: only SceneEntityCfg objects inside params get resolved
        # against the scene, so a default one would still hold body_ids=slice(None).
        body_friction = ObsTerm(
            func=body_material_friction,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            # No history here on purpose: history exists so the actor can infer
            # what it cannot measure, and the critic already measures it
            # directly (base_lin_vel, wrench, friction). Stacking it would cost
            # memory and bandwidth for nothing.

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Resets plus Week04's domain randomization (FR-T5).

    Five items, exactly the five the plan asks for and no more: friction,
    mass, motor strength, control delay and external pushes. Four of them are
    events; the fifth (control delay) is not expressible as an event at all
    and lives in the action term -- see ActionsCfg above.

    Note ``randomize_actuator_gains`` runs at ``startup``, not per reset. R1's
    actuators are ``ImplicitActuatorCfg``, and Isaac Lab's own note on that
    function says implicit actuators are written through CPU tensors, so it is
    "recommended to use this function only during the initialization of the
    environment". Per-reset randomization of 4096 envs through the CPU would
    dominate the step time. Startup-only means each env keeps one fixed motor
    strength for the whole run, which is the right model anyway: a real motor's
    strength doesn't resample when the robot gets picked up.

    Ranges are deliberately narrow-to-moderate. The Week04 plan's own risk note
    is "DR 开太猛会导致学不动 -> 从窄区间起，逐步放宽" -- this is the starting
    width, to be widened only after a run confirms the gait survives it.
    """

    # -- domain randomization (FR-T5) --
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.6, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

    # -- resets --
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
    """Week03 reward set (FR-T3): "tracking is positive, everything else is a
    penalty regularizer", per the plan.

    Revision history:
    - First training (run `2026-08-14_14-22-09_first_train`): literature-standard
      H1-template weights. Result: reward rose -5.2 -> 9.75 and R1 stopped
      falling, but `scripts/play_r1.py` showed a degenerate gait -- legs
      splayed wide, feet dragging/skating instead of stepping. tensorboard
      showed why: `Episode_Reward/feet_air_time` stayed ~0 for the *entire*
      run (never a live gradient) while `joint_deviation_hip` (hip_roll+yaw
      combined) settled into a bad value by iteration ~300-450 and never
      recovered -- the wide-stance drag was a cheaper local optimum (lower
      torque/accel/action-rate) than real stepping, and nothing was strong
      enough to outweigh that. Full diagnosis:
      ~/kdw/experiment_record/Week03_首训_宽步态与奖励回落问题.md
    - This revision (`reward_fix`): raised feet_air_time and feet_slide, split
      hip deviation so hip_roll (the "splay" DOF) is penalized much harder
      than hip_yaw (which doesn't cause splay), and added base_height_l2 to
      anchor posture near Week02's verified standing height instead of letting
      the policy crouch/widen further for cheap stability. Deliberately did
      NOT raise dof_torques_l2/dof_acc_l2/action_rate_l2 -- those already
      favor the low-effort drag solution, raising them would fight the fix.
    - This revision (`reward_fix_symmetry`, Option A, v1 -- FAILED): reward_fix
      fixed the wide-drag gait but exposed a second, subtler exploit -- one
      leg walks normally while the other stays mostly planted and just taps
      up/down briefly, since `feet_air_time_positive_biped` only requires
      momentary single-foot-support and can't distinguish a real swing phase
      from a tap. Confirmed structural (not undertraining) via an unmodified
      reward_fix control run to 1500 iterations, where the relevant metrics
      plateaued by ~iteration 800 with the asymmetry still visible on replay.
      Added a penalty on left/right *instantaneous* air-time-duty-fraction
      difference (`current_air_time`/`current_contact_time`-based). Trained
      and visually verified: FAILED -- this formula penalizes *any*
      single-stance instant almost as hard as a real asymmetric tap (the
      swinging foot's instantaneous duty is always ~1, the planted foot's
      ~0), so it fought feet_air_time_positive_biped's own incentive and the
      policy learned to avoid single-stance entirely (both feet moving in a
      synced hop/shuffle instead of alternating steps). Diagnosis:
      ~/kdw/experiment_record/Week03_reward_fix_symmetry_方案A失败复盘与v2方案.md
    - This revision (`reward_fix_symmetry_v2` -- ALSO FAILED): fixed v1's
      formula bug by comparing `last_air_time` (most recently *completed*
      swing) instead of the instantaneous duty fraction. Looked like a
      success on tensorboard, but `scripts/diagnose_gait.py` (written after
      the user reported the gait still looked wrong) measured the truth:
      left foot airborne 5.6% of the time, right foot 94.9% -- R1 was
      hopping along on its left leg with the right leg held up. The control
      run with no symmetry penalty at all measured 6.6%/94.9%, i.e. all
      three symmetry rounds had been attacking a symptom.
    - This revision (`touchdown_gate`): fixes the actual root cause.
      `feet_air_time_positive_biped` is *maximized* by standing on one leg
      (permanent single-stance keeps both feet's in_mode_time growing, so
      min(...) sits clamped at threshold forever -- never stepping is that
      term's global optimum, and it had weight 1.0 since reward_fix).
      Replaced it with `feet_swing_touchdown` (tasks/r1_flat/mdp.py), which
      settles credit only when a foot actually lands, and dropped the
      useless symmetry term. Added `feet_air_time_excess` to penalize a leg
      held up past 0.6s every frame it stays up. Diagnosis + the gait-phase
      design held in reserve if this isn't enough:
      ~/kdw/experiment_record/Week03_单腿支撑漏洞根因与步态相位方案.md

    -- task (positive) --
    track_lin_vel_xy_exp / track_ang_vel_z_exp: command tracking, exp kernel.
    alive: small constant per-step reward. Without it, the only signal near a
        fall is the (large, sparse) termination penalty -- alive gives a dense
        gradient against "learn to fall over slowly" degenerate solutions.
    feet_swing_touchdown: rewards each *completed* swing at the moment the
        foot lands (replaces feet_air_time_positive_biped, which paid every
        frame and was therefore maximized by never landing at all). Weight
        10.0 looks large next to the others but the signal is sparse -- it
        pays on ~2 frames per gait cycle instead of all 40, so its average
        per-step contribution lands around 0.15 for a healthy gait vs
        tracking's ~0.9. Do NOT compare this term's tensorboard values
        against the old feet_air_time's: different function, different
        units, and the old one read *higher* the worse the gait got.

    -- physical plausibility (penalty, task-agnostic) --
    lin_vel_z_l2 / ang_vel_xy_l2 / flat_orientation_l2: vertical bounce, tilt
        rate, and tilt angle -- flat_orientation_l2 penalizes the angle
        itself; ang_vel_xy_l2 only penalizes its rate, so the two are
        complementary, not redundant.
    base_height_l2: new (reward_fix) -- anchors pelvis height near 0.72m (the
        Week02-verified standing height) so the policy can't trade a lower/
        wider crouch for cheap stability instead of learning to step.
    dof_torques_l2 / dof_acc_l2 / action_rate_l2: energy + smoothness, keeps
        the policy off high-frequency torque exploits that don't transfer.
        Deliberately unchanged in reward_fix -- see revision history above.
    dof_pos_limits: penalizes legs running into soft joint limits (the actual
        moving joints during gait; arms/waist are handled by joint_deviation
        below instead, since they shouldn't be moving much at all).
    feet_slide: penalizes foot velocity while in contact (skating instead of
        a clean plant/lift). weight -0.25 -> -1.0 (reward_fix): the dragging
        gait's feet_slide penalty was consistently small (~-0.03 to -0.06),
        too cheap relative to what dragging saved elsewhere.
    feet_air_time_excess: penalizes a foot staying airborne past 0.6s, every
        frame it stays up. Inert for normal swings (~0.3s); its whole job is
        to make "hold a leg in the air" actively expensive rather than
        merely unrewarded. Fires continuously, so unlike the abandoned
        symmetry terms it doesn't depend on a landing that never comes.

    -- upper-body regularization (penalty, R1-specific) --
    joint_deviation_arms / joint_deviation_waist: R1 has 14 non-leg DOF with
        no task of their own this week. Without a "stay near default" pull,
        the policy can use arm/waist flailing as a free balance aid in sim --
        doesn't transfer to a real robot and fights a future arm-task's prior.
    joint_deviation_hip_roll / joint_deviation_hip_yaw: split apart in
        reward_fix (was one combined joint_deviation_hip term at -0.1). Wide
        stance is specifically a hip_roll (abduction) problem, not hip_yaw --
        combining them let hip_roll deviate as long as the *sum* stayed
        moderate. hip_roll now -0.4; hip_yaw stays -0.1 (still relevant since
        this week's commands are straight-line-only, no turning).

    termination_penalty: large one-time penalty on a non-timeout termination
        (falling), separate from and much larger than the per-step alive
        reward -- makes "don't fall" dominate the return even though alive
        accumulates every step.
    """

    # -- task --
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    feet_swing_touchdown = RewTerm(
        func=feet_swing_touchdown,
        weight=10.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "min_swing_time": 0.15,
            "threshold": 0.45,
        },
    )

    # -- physical plausibility --
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=-1.0, params={"target_height": 0.731})
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"]
            )
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_air_time_excess = RewTerm(
        func=feet_air_time_excess_l1,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "max_air_time": 0.6,
        },
    )

    # -- upper-body regularization --
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_roll_joint"]
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint", "waist_roll_joint"])},
    )
    joint_deviation_hip_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])},
    )
    joint_deviation_hip_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"])},
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class TerminationsCfg:
    """"摔倒": base height too low, or tilted too far. The height threshold
    tracks the default standing pelvis height (0.731m as of the Week04 actuator
    correction, see assets/r1/r1.py) at the same ~0.69 ratio the 0.5/0.72 pair
    used, so "fallen" keeps meaning the same posture rather than drifting when
    the nominal pose changes."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.51})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.7})


@configclass
class CurriculumCfg:
    """Week04 command curriculum.

    Week03 produced a policy that walks well at 0-0.5 m/s straight ahead.
    Opening the full 0-1.0 m/s range plus turning from step 0 throws that head
    start away; ramping in keeps it. ``end_step`` is in *control* steps --
    rsl_rl runs 24 per iteration, so 24000 steps is iteration 1000, i.e. the
    ranges are fully open for the last two thirds of a 3000-iteration run.
    """

    command_ranges = CurrTerm(
        func=command_range_curriculum,
        params={
            "command_name": "base_velocity",
            "initial_ranges": COMMAND_RANGES_INITIAL,
            "final_ranges": COMMAND_RANGES_FINAL,
            "start_step": 0,
            "end_step": 24000,
        },
    )


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
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # 200Hz physics, 50Hz control -- the Isaac Lab locomotion default.
        #
        # This was dt=0.002 / decimation=10 (500Hz physics) through Week04, and
        # that was never really about R1: it was forced by the Week02 leg
        # stiffness of 1200 N*m/rad, at which PhysX's implicit drive solver
        # diverged at coarser timesteps. Those gains have since been corrected
        # to the hardware's own 100/40 (assets/r1/r1.py), which removes the
        # constraint and gives back 2.5x of physics cost. Verified stable at
        # these gains before adopting; revert to 0.002/10 if a future gain
        # change reintroduces stiffness in that range.
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class R1FlatEnvCfg_PLAY(R1FlatEnvCfg):
    """Small/deterministic variant for interactive play/verification runs.

    Week04: domain randomization and the command curriculum are switched off
    here so that play/evaluation runs at *nominal* conditions, with the full
    command range available immediately. Otherwise every measurement would
    carry a random friction/mass/gain draw, and scripts/eval_baseline.py could
    only ever command the curriculum's step-0 range. Stress conditions are
    applied explicitly instead, via that script's --friction / --push_vel.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        # nominal conditions, not a random draw
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.actuator_gains = None
        self.events.push_robot = None
        self.actions.joint_pos.max_delay = 0
        # full command range from the start, no ramp
        self.curriculum.command_ranges = None
        self.commands.base_velocity.ranges.lin_vel_x = COMMAND_RANGES_FINAL["lin_vel_x"]
        self.commands.base_velocity.ranges.ang_vel_z = COMMAND_RANGES_FINAL["ang_vel_z"]
