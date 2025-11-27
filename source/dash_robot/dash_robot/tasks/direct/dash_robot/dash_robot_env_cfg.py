# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.dash import DASH_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class DashRobotEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0

    # - spaces definition
    action_space = 18 # 18 joints
    observation_space = 69
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    

    # robot(s)
    robot_cfg: ArticulationCfg = DASH_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    joint_names: list = [
        "l_hip_yaw", 
        "l_hip_roll", 
        "l_hip_pitch", 
        "l_knee_pitch", 
        "l_ankle_pitch", 
        "l_shoulder_pitch",
        "l_shoulder_roll",
        "l_shoulder_yaw",
        "l_elbow_pitch",
        "r_hip_yaw",
        "r_hip_roll",
        "r_hip_pitch",
        "r_knee_pitch",
        "r_ankle_pitch",
        "r_shoulder_pitch",
        "r_shoulder_roll",
        "r_shoulder_yaw",
        "r_elbow_pitch"
    ]
    # - action scale
    action_scale = 100.0  # [N]

    # Termination thresholds, example values for now (estimated)
    min_torso_height: float = 0.7
    min_torso_up: float = 0.2

    # - Reward scales
    rew_scale_alive = 2.0
    rew_scale_terminated = -2.0
    rew_joint_vel = 0.1