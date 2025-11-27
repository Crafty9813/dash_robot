import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# from source.whole_body_tracking.whole_body_tracking.assets import ASSET_DIR

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

DASH_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        usd_path="/home/Repositories/dash_robot/source/assets/robot.usd",
        
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
            # enabled_self_collisions=False,
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_hip_yaw": 0.0,
            ".*_hip_roll": 0.0,
            ".*_hip_pitch": 0,  # -16 degrees
            ".*_knee_pitch": 0,  # 45 degrees
            ".*_ankle_pitch": 0,  # -30 degrees
            ".*_shoulder_pitch": 0.0,  # -30 degrees
            ".*_shoulder_roll": 0.0,  # -30 degrees
            ".*_shoulder_yaw": 0.0,  # -30 degrees
            ".*_elbow_pitch": 0.0,  # -30 degrees
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw",
                ".*_hip_roll",
                ".*_hip_pitch",
                ".*_knee_pitch",
            ],
            effort_limit_sim={
                ".*_hip_yaw": 88.0,
                ".*_hip_roll": 139.0,
                ".*_hip_pitch": 88.0,
                ".*_knee_pitch": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw": 32.0,
                ".*_hip_roll": 20.0,
                ".*_hip_pitch": 32.0,
                ".*_knee_pitch": 20.0,
            },
            # effort_limit_sim={
                # ".*_hip_yaw": 1000.0,
                # ".*_hip_roll": 1000.0,
                # ".*_hip_pitch": 1000.0,
                # ".*_knee_pitch": 1000.0,
            # },
            # velocity_limit_sim={
                # ".*_hip_yaw": 1000.0,
                # ".*_hip_roll": 1000.0,
                # ".*_hip_pitch": 1000.0,
                # ".*_knee_pitch": 1000.0,
            # },
            stiffness={
                ".*_hip_yaw": STIFFNESS_7520_14,
                ".*_hip_roll": STIFFNESS_7520_22,
                ".*_hip_pitch": STIFFNESS_7520_14,
                ".*_knee_pitch": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_yaw": DAMPING_7520_14,
                ".*_hip_roll": DAMPING_7520_22,
                ".*_hip_pitch": DAMPING_7520_14,
                ".*_knee_pitch": DAMPING_7520_22,
            },
            armature={
                ".*_hip_yaw": ARMATURE_7520_14,
                ".*_hip_roll": ARMATURE_7520_22,
                ".*_hip_pitch": ARMATURE_7520_14,
                ".*_knee_pitch": ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=50,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        # "feet": ImplicitActuatorCfg(
            # effort_limit_sim=1000,
            # velocity_limit_sim=1000.0,
            # joint_names_expr=[".*_ankle_pitch"],
            # stiffness=2.0 * STIFFNESS_5020,
            # damping=2.0 * DAMPING_5020,
            # armature=2.0 * ARMATURE_5020,
        # ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch",
                ".*_shoulder_roll",
                ".*_shoulder_yaw",
                ".*_elbow_pitch",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch": 25.0,
                ".*_shoulder_roll": 25.0,
                ".*_shoulder_yaw": 25.0,
                ".*_elbow_pitch": 25.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch": 37.0,
                ".*_shoulder_roll": 37.0,
                ".*_shoulder_yaw": 37.0,
                ".*_elbow_pitch": 37.0,
            },
            stiffness={
                ".*_shoulder_pitch": STIFFNESS_5020,
                ".*_shoulder_roll": STIFFNESS_5020,
                ".*_shoulder_yaw": STIFFNESS_5020,
                ".*_elbow_pitch": STIFFNESS_5020,
            },
            damping={
                ".*_shoulder_pitch": DAMPING_5020,
                ".*_shoulder_roll": DAMPING_5020,
                ".*_shoulder_yaw": DAMPING_5020,
                ".*_elbow_pitch": DAMPING_5020,
            },
            armature={
                ".*_shoulder_pitch": ARMATURE_5020,
                ".*_shoulder_roll": ARMATURE_5020,
                ".*_shoulder_yaw": ARMATURE_5020,
                ".*_elbow_pitch": ARMATURE_5020,
            },
        ),
    },
)
"""Configuration of PresToe robot."""

DASH_ACTION_SCALE = {}
for a in DASH_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            DASH_ACTION_SCALE[n] = 0.25 * e[n] / s[n]