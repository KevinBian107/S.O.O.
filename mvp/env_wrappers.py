import os
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal
import torch.nn as nn

class TargetVelocityWrapper(gym.Wrapper):
    def __init__(self, env, target_velocity=2.0, tolerance=0.5):
        super(TargetVelocityWrapper, self).__init__(env)
        self.target_velocity = target_velocity
        self.tolerance = tolerance

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        velocity = obs[8]  # Assuming velocity is part of the observation (e.g., index 8)
        
        # Calculate how close the velocity is to the target velocity
        velocity_error = abs(self.target_velocity - velocity)
        velocity_reward = max(0, 1 - (velocity_error / self.tolerance))  # Higher reward for being close to target

        # Modify the reward based on velocity proximity to the target
        reward = velocity_reward

        return obs, reward, terminated, truncated, info
    
class JumpRewardWrapper(gym.Wrapper):
    def __init__(self, env, jump_target_height=1.0):
        super(JumpRewardWrapper, self).__init__(env)
        self.jump_target_height = jump_target_height

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        torso_height = obs[0]  # Assuming the torso's z-coordinate is at index 0 (check observation space)

        # Reward based on how high the torso is, encouraging jumps
        height_reward = torso_height / self.jump_target_height

        reward = height_reward  # Override original reward with height-based reward

        return obs, reward, terminated, truncated, info