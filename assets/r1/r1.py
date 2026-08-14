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
        # Shared nominal posture — keep consistent with the R1 model's home/keyframe
        # pose (mjcf/R1_C++.xml) 
        # Moderate-crouch, bent-arm pose.
        # pelvis z = 0.72 puts the soles flat on the ground for this leg posture:
        # URDF leg FK gives ankle_roll_link at -0.66675 m below pelvis, plus the
        # 0.053245 m sole offset (robot.yaml contact_plane_offset_z) => 0.72.
        pos=(0.0, 0.0, 0.72),
        joint_pos={
            # Legs: moderate flat-foot crouch (hip + knee + ankle = 0 → soles flat).
            ".*_hip_pitch_joint": -0.26,
            ".*_knee_joint": 0.52,
            ".*_ankle_pitch_joint": -0.26,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_ankle_roll_joint": 0.0,
            # Waist.
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            # Arms: relaxed natural bend (slight forward + abduction + bent elbow)
            # so the idle pose looks human; matches the TSID posture-task reference.
            ".*_shoulder_pitch_joint": 0.15,
            "left_shoulder_roll_joint": 0.10,
            "right_shoulder_roll_joint": -0.10,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.35,
            ".*_wrist_roll_joint": 0.0,
            # head
            "head_pitch_joint": 0.0,
            "head_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Modeled actuators.
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim={
                ".*_hip_.*_joint": 150.0,
                ".*_knee_joint": 150.0,
                ".*_ankle_.*_joint": 150.0,
            },
            stiffness={
                # Week02 standing check (no whole-body balance controller --
                # pure joint-PD hold), full diagnosis after 10 real Isaac Sim
                # iterations:
                #
                # 1) Ruled out a bad default pose: CoM sits only 2.6cm forward
                #    of the ankle, well inside the foot's 13.5cm forward margin
                #    (mesh bounds) -- not a static support-polygon violation.
                # 2) GUI observation: R1 fell forward *rigidly* (no joint
                #    buckling) -- a whole-body inverted-pendulum problem, not
                #    per-joint compliance.
                # 3) Quantitatively: modeling the robot as an inverted pendulum
                #    (mass~28.8kg, CoM height~0.67m) gives a destabilizing
                #    gravity "spring constant" of m*g*h ~= 190 N*m/rad. The leg
                #    chain's effective stiffness at the CoM is hip/knee/ankle
                #    combined *in series* like springs, dominated by the
                #    softest joint -- every attempt with hip <=250 stayed below
                #    the 190 threshold regardless of ankle stiffness, which is
                #    why raising ankle alone (up to 30x) never helped.
                # 4) Raising all three past the threshold (this stiffness) via
                #    the *default* physics timestep (1/60s) STILL fell,
                #    identically. Root cause: at ~1200 N*m/rad, 1/60s is too
                #    coarse for PhysX's implicit joint-drive solver -- it's a
                #    numerical instability, not a physical one. Fixed by the
                #    sim dt, not the gains (see scripts/inspect_r1.py and
                #    tasks/r1_flat/flat_env_cfg.py, both now dt=0.002).
                #
                # Both (3) and (4) were necessary; neither alone was sufficient.
                ".*_hip_.*_joint": 1200.0,
                ".*_knee_joint": 1200.0,
                ".*_ankle_.*_joint": 1200.0,
            },
            damping={
                ".*_hip_.*_joint": 100.0,
                ".*_knee_joint": 100.0,
                ".*_ankle_.*_joint": 150.0,
            },
            armature=0.01,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "waist_roll_joint"],
            effort_limit_sim=60.0,
            # Raised 200/5 -> 400/20 (Week02): the waist carries the whole
            # upper body (torso+arms+head, waist_yaw_link alone is 6kg, the
            # single heaviest body, mounted well above the pelvis) on what
            # was low damping relative to that load. Untested lever after leg
            # gains alone (up to 30x) didn't stop a standing collapse whose
            # timing (~1.5s) barely changed across very different leg gains --
            # suggesting the drift wasn't coming from the legs at all.
            stiffness=400.0,
            damping=20.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim=40.0,
            stiffness=80.0,
            damping=15.0,
            armature=0.01,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_pitch_joint", "head_yaw_joint"],
            effort_limit_sim=33.0,
            stiffness=40.0,
            damping=10.0,
            armature=0.01,
        ),
    },
)
"""Configuration for the R1 humanoid robot (USD generated from urdf/R1.urdf)."""
