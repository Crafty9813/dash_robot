# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.utils.math import quat_apply

from .dash_robot_env_cfg import DashRobotEnvCfg


class DashRobotEnv(DirectRLEnv):
    cfg: DashRobotEnvCfg

    def __init__(self, cfg: DashRobotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.joint_ids, _ = self.robot.find_joints(self.cfg.joint_names)
        self.num_joints = len(cfg.joint_names)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel


    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(self.actions * self.cfg.action_scale, joint_ids=self.joint_ids)

    # Based on joint positions and velocities
    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self.joint_ids],
                self.joint_vel[:, self.joint_ids],
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        torso_height = self.robot.data.root_pos_w[:, 2]
        torso_up = self.robot.data.root_quat_w[:, 2]

        '''
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.joint_pos[:, self.joints[0]],
            self.joint_vel[:, self.joints[0]],
            self.reset_terminated,
        )'''

        total_reward = compute_rewards(
            torso_height,
            torso_up,
            self.joint_vel[:, self.joint_ids],
            self.cfg.min_torso_height,
            self.cfg.min_torso_up,
            self.cfg.rew_alive,
            self.cfg.rew_fall,
            self.cfg.rew_joint_vel,
            self.reset_terminated,
        )
        return total_reward

    # Terminate if fall
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        torso_height = self.robot.data.root_pos_w[:, 2]

        z_axis = torch.tensor([0, 0, 1], device=self.device)
        torso_up_vec = quat_apply(self.robot.data.root_quat_w, z_axis)
        torso_up = torso_up_vec[:, 2] # Measures how upright the robot is

        fallen = (torso_height < self.cfg.min_torso_height) | (torso_up < self.cfg.min_torso_up)

        timeout = self.episode_length_buf >= self.max_episode_length - 1

        return fallen, timeout

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        # Noise so starting pose different each time?
        joint_pos += 0.01 * torch.randn_like(joint_pos)
        joint_vel += 0.01 * torch.randn_like(joint_vel)

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    torso_height: torch.Tensor,
    torso_up: torch.Tensor,
    joint_vel: torch.Tensor,
    min_height: float,
    min_up: float,
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_joint_vel: float,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()

    # Encourage joint movement
    joint_vel_reward = rew_joint_vel * torch.sum(joint_vel * joint_vel, dim=1)

    total_reward = rew_alive + rew_termination + joint_vel_reward
    return total_reward