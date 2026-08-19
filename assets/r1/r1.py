# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the R1 humanoid robot.

The R1 USD is generated *from* ``assets/r1/R1.urdf`` via
``scripts/convert_r1_urdf.py``. The simulated robot and the
TSID/Pinocchio model are therefore the same robot, so the foot/leg/mass geometry
matches and the standing tilt caused by a sim-vs-model mismatch is eliminated.

26 actuated DOF: 12 legs, 2 waist, 10 arms, 2 head. Feet are ``*_ankle_roll_link``.
The parallel-ankle rods (``*_ankle_A/B*``) are fixed joints in the URDF and are
lumped into their parent by both the USD converter and Pinocchio.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# All R1 model files live in the package next to this file: assets/r1/. R1.urdf
# and meshes/ are tracked in git; usd/ is a *build artefact* produced by
# scripts/convert_r1_urdf.py and is git-ignored. Keeping the USD out of version
# control means the URDF stays the single source of truth: a URDF edit cannot
# silently disagree with a stale USD that someone converted months ago, which is
# exactly the sim-versus-model mismatch this asset pipeline exists to avoid.
R1_ROOT = Path(__file__).resolve().parent
R1_URDF_PATH = str(R1_ROOT / "R1.urdf")
R1_USD_PATH = str(R1_ROOT / "usd" / "R1.usd")


def require_usd() -> str:
    """Return the R1 USD path, with a build instruction if it has not been generated.

    Called at spawn time rather than import time so the package still imports on
    a machine without Isaac Lab.
    """
    if not Path(R1_USD_PATH).exists():
        raise FileNotFoundError(
            f"R1 USD not found at {R1_USD_PATH}.\n"
            "The USD is a build artefact generated from the tracked URDF. Run:\n"
            "    ~/IsaacLab/isaaclab.sh -p scripts/convert_r1_urdf.py\n"
            "(from the R1process project root). See scripts/README.md."
        )
    return R1_USD_PATH


R1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=R1_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Unitree's official HOME_KEYFRAME for this robot (unitree_rl_mjlab's
        # r1_constants.py). Adopted together with their actuator gains below --
        # the two go together: the previous pose was a deeper crouch
        # (z=0.72, hip -0.26 / knee 0.52 / ankle -0.26) which needs noticeably
        # more static holding torque, and that is exactly the budget the soft
        # official gains no longer have.
        #
        # Mirror-symmetric by construction (only shoulder_roll differs left vs
        # right, and it differs by sign) -- required by the symmetry
        # augmentation in tasks/r1_flat/symmetry.py, which mirrors joint_pos
        # *relative to this default*.
        # NOTE the height is *ours*, not Unitree's 0.76. Their joint angles
        # transfer directly (hip -0.1 + knee 0.3 + ankle -0.2 = 0, so the soles
        # stay horizontal), but the pelvis height that puts those soles on the
        # ground depends on foot collision geometry, which differs between their
        # MJCF and this URDF. Measured here: leg FK puts ankle_roll_link 0.6778m
        # below the pelvis, plus the 0.053245m sole offset (robot.yaml
        # contact_plane_offset_z) => 0.731. Spawning at 0.76 left the feet
        # floating 2.9cm and dropping on every reset.
        pos=(0.0, 0.0, 0.731),
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_ankle_roll_joint": 0.0,
            # Waist.
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            # Arms.
            ".*_shoulder_pitch_joint": 0.35,
            "left_shoulder_roll_joint": 0.18,
            "right_shoulder_roll_joint": -0.18,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.87,
            ".*_wrist_roll_joint": 0.0,
            # Head (not covered by Unitree's config -- their MJCF doesn't
            # actuate it).
            "head_pitch_joint": 0.0,
            "head_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Actuator parameters follow the robot's own hardware spec and Unitree's
        # official RL configuration, which agree with each other exactly:
        #   - effort limits are what `R1.urdf` declares (<limit effort=...>):
        #     60 N*m hip/knee/waist/shoulder, 50 N*m ankle, 33 N*m wrist-group;
        #   - stiffness/damping/armature are unitree_rl_mjlab's
        #     src/assets/robots/unitree_r1/r1_constants.py.
        #
        # These replaced a much stiffer, much stronger set carried over from
        # Week02 (legs 1200/100, ankle 1200/150, all leg efforts 150 N*m). That
        # was a mistake with a specific cause, worth recording because the
        # reasoning looked sound at the time:
        #
        #   Week02's criterion was "hold the default pose under *pure joint PD*,
        #   no controller, for >=10s", which needs the leg chain's passive
        #   stiffness to beat the inverted pendulum's m*g*h ~= 190 N*m/rad.
        #   That is the wrong criterion for an RL task: the policy re-targets
        #   every joint at 50Hz and balances *actively*, so it never needs the
        #   pose to be passively self-supporting. Unitree ships 100/40 on this
        #   same robot precisely because the policy does the balancing.
        #
        # Everything downstream came from that one choice: stiffness 1200 made
        # PhysX's implicit drive solver diverge at 1/60s (hence sim dt=0.002),
        # and made explicit actuators unusable entirely (Week04's
        # DelayedPDActuatorCfg smoke test collapsed in 1.5s). Both constraints
        # relax at these gains -- explicit-PD damping stability wants roughly
        # dt < 2J/d, which is ~0.01s here instead of ~1e-4s.
        #
        # The measured cost of the old values: the Week04 policy spent 11.6% of
        # its time commanding ankle torques above the real R1's 50 N*m rating,
        # with p99 pinned at the 150 N*m sim ceiling -- a gait no hardware could
        # reproduce. See ~/kdw/experiment_record/Week04_执行器参数与硬件规格不符_根因与修正方案.md
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            effort_limit=60.0,
            effort_limit_sim=60.0,
            stiffness=100.0,
            damping=2.0,
            armature=0.01,
        ),
        # Split out from "legs": the ankles are softer and weaker than the rest
        # of the leg on the real robot (40/2 and 50 N*m), and lumping them in
        # with the hips is what let the old config drive them at 3x their rating.
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit=50.0,
            effort_limit_sim=50.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "waist_roll_joint"],
            effort_limit=60.0,
            effort_limit_sim=60.0,
            stiffness=100.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"],
            effort_limit=60.0,
            effort_limit_sim=60.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit=33.0,
            effort_limit_sim=33.0,
            stiffness=20.0,
            damping=1.0,
            armature=0.01,
        ),
        # Unitree's config has no head group (their MJCF doesn't actuate it), so
        # this one is ours: the URDF rates these joints at 33 N*m, and they sit
        # in the same size class as the wrist group, so they get its gains.
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_pitch_joint", "head_yaw_joint"],
            effort_limit=33.0,
            effort_limit_sim=33.0,
            stiffness=20.0,
            damping=1.0,
            armature=0.01,
        ),
    },
)
"""Configuration for the R1 humanoid robot (USD generated from urdf/R1.urdf)."""
