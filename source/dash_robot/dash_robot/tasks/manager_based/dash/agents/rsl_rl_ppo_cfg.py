# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
#from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symmetry import dash

'''
def mirror_joint_tensor(original: torch.Tensor, mirrored: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """Mirror a tensor of joint values by swapping left/right pairs and inverting yaw/roll joints.
    
    Args:
        original: Input tensor of shape [..., num_joints] where num_joints is 23
        mirrored: Output tensor of same shape to store mirrored values
        offset: Optional offset to add to indices if tensor has additional dimensions
        
    Returns:
        Mirrored tensor with same shape as input
    """
    # Define pairs of indices to swap (left/right pairs)
    swap_pairs = [
        (2 + offset, 11 + offset),   # hip_pitch
        (1 + offset, 10 + offset),   # hip_roll
        (0 + offset, 9 + offset),   # hip_yaw
        (3 + offset, 12 + offset),   # knee
        (5 + offset, 14 + offset),  # shoulder_pitch
        (4 + offset, 13 + offset), # ankle_pitch
        (6 + offset, 15 + offset), # shoulder_roll
        (7 + offset, 16 + offset), # shoulder_yaw
        (8 + offset, 17 + offset), # elbow
    ]
    
    # Define indices that need to be inverted (yaw/roll joints)
    invert_indices = [
        2 + offset,   # waist_yaw
        1 + offset,   # left_hip_roll
        10 + offset,   # right_hip_roll
        0 + offset,   # left_hip_yaw
        9 + offset,   # right_hip_yaw
        6 + offset,  # left_shoulder_roll
        15 + offset,  # right_shoulder_roll
        7 + offset,  # left_shoulder_yaw
        16 + offset,  # right_shoulder_yaw
    ]

    non_swap_indices = [i for i in range(original.shape[-1]) if i not in [idx for pair in swap_pairs for idx in pair]]
    mirrored[..., non_swap_indices] = original[..., non_swap_indices]
    
    # Swap left/right pairs
    for left_idx, right_idx in swap_pairs:
        mirrored[..., left_idx] = original[..., right_idx]
        mirrored[..., right_idx] = original[..., left_idx]
    
    # Invert yaw/roll joints
    mirrored[..., invert_indices] = -mirrored[..., invert_indices]

def mirror_policy_observation(policy_obs):
    # Assume policy_obs has [..., qpos, qvel, ...], mirror relevant parts
    # Example: Mirroring joint angles (qpos) and velocities (qvel)
    mirrored_qpos = policy_obs[..., 3:13].clone() # Example indices for legs/arms
    mirrored_qpos[..., :3] *= -1 # Flip X (side) and Y (front/back)
    mirrored_qpos[..., 3:6] *= -1 # Flip if necessary
    # Add more specific mirroring for other observations (IMU, sensors)

    mirrored_obs = policy_obs.clone()
    mirrored_obs[..., 3:13] = mirrored_qpos # Place mirrored data back
    return mirrored_obs

def mirror_actions(actions):
    # Mirror action commands (e.g., torques, positions)
    mirrored_actions = actions.clone()
    # Example: Flip left/right action commands
    mirrored_actions[..., 0:3] *= -1 # Example for left arm/leg actions
    mirrored_actions[..., 3:6] *= -1 # Example for right arm/leg actions
    return mirrored_actions

def data_augmentation_func_dash(env, obs, actions, obs_type):
    if obs_type == "policy":
        obs_batch = mirror_observation_policy(obs)
    elif obs_type == "critic":
        obs_batch = mirror_observation_critic(obs)
    else:
        raise ValueError(f"Invalid observation type: {obs_type}")
    
    mean_actions_batch = mirror_actions(actions)
    return obs_batch, mean_actions_batch'''

@configclass
class DashFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 50
    experiment_name = "test"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0
    )


'''
@configclass
class DashFlatPPORunnerWithSymmetryCfg(DashFlatPPORunnerCfg):
    """Configuration for the PPO agent with symmetry augmentation."""

    # all the other settings are inherited from the parent class
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func=dash.compute_symmetric_states
        ),
    )


@configclass
class DashRoughPPORunnerWithSymmetryCfg(DashFlatPPORunnerCfg):
    """Configuration for the PPO agent with symmetry augmentation."""

    # all the other settings are inherited from the parent class
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func=dash.compute_symmetric_states
        ),
    )'''
