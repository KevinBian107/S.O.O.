import gymnasium as gym
import numpy as np
from gymnasium.experimental.wrappers.rendering import RecordVideoV0 as RecordVideo
from env.env_wrappers import *

def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = RecordVideo(env, f"videos/{run_name}")
            # fixed it by reading Stack Overfloat
        else:
            env = gym.make(env_id)

        # env = TargetVelocityWrapper(env, target_velocity=2.0)
        # env = JumpRewardWrapper(env, jump_target_height=2.0)
        # env = PartialObservabilityWrapper(env=env, observable_ratio=0.2)
        # env = ActionMaskingWrapper(env=env, mask_prob=0.2)
        # env = DelayedRewardWrapper(env, delay_steps=20)
        # env = NonLinearDynamicsWrapper(env, dynamic_change_threshold=50)
        # env = NoisyObservationWrapper(env, noise_scale=0.1)
        # env = DelayedHalfCheetahEnv(env=env, proprio_delay=1, force_delay=3)
        
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def make_env_with_wrapper(env_id, idx, capture_video, run_name, gamma, wrappers=None):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = RecordVideo(env, video_folder=f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        if wrappers is not None:
            for wrapper_fn in wrappers:
                env = wrapper_fn(env)
                
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

        return env

    return thunk
