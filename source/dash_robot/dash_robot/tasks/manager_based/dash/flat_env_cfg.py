# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import DashRoughEnvCfg

@configclass
class DashFlatEnvCfg(DashRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        #self.observations.policy.velocity_commands = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards
        #self.rewards.joint_t_pose_l2.weight = -1.0
        
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.1
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 3.5
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_pitch"]
        )

class DashFlatEnvCfg_PLAY(DashFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
