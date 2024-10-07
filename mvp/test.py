import os
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal
import torch.nn as nn
from fmppo_vector import Args
from env_wrappers import JumpRewardWrapper, TargetVelocityWrapper
    
# UPN model definition
class UPN(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim):
        super(UPN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state, action, next_state):
        z = self.encoder(state)
        z_next = self.encoder(next_state)
        z_pred = self.dynamics(torch.cat([z, action], dim=-1))
        action_pred = self.inverse_dynamics(torch.cat([z, z_next], dim=-1))
        state_recon = self.decoder(z)
        next_state_pred = self.decoder(z_pred)
        return z, z_next, z_pred, action_pred, state_recon, next_state_pred

# Agent class for PPO
class PPOAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_action(self, x):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = probs.sample()
        return action

# Agent class for FM-PPO
class FMPPOAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        state_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)
        latent_dim = 32

        self.upn = UPN(state_dim, action_dim, latent_dim)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_action(self, x):
        z = self.upn.encoder(x)
        action_mean = self.actor_mean(z)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = probs.sample()
        return action

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def evaluate_model(agent, env, device, num_episodes=100):
    returns = []

    for episode in range(num_episodes):
        print(f'Episode: {episode}')
        obs, _ = env.reset()
        obs = torch.Tensor(obs).to(device)
        done = False
        episode_return = 0

        while not done:
            with torch.no_grad():
                action = agent.get_action(obs)
                obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
                obs = torch.Tensor(obs).to(device)
                episode_return += reward
                done = terminated or truncated

        returns.append(episode_return)

    return returns

if __name__ == "__main__":
    args = Args()
    # Set up vectorized environment with a single instance
    env_id = "HalfCheetah-v4"
    # env = gym.vector.SyncVectorEnv([lambda: TargetVelocityWrapper(gym.make(env_id), target_velocity=1.0)])
    # env = gym.vector.SyncVectorEnv([lambda: gym.make(env_id)])
    env = gym.vector.SyncVectorEnv([lambda: JumpRewardWrapper(gym.make(env_id), jump_target_height=1.0)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the FM-PPO model
    fmppo_agent = FMPPOAgent(env).to(device)
    fmppo_path = os.path.join(os.getcwd(), "mvp", "params", "fmppo_vector.pth")
    fmppo_agent.load_state_dict(torch.load(fmppo_path, map_location=device))

    # Load the PPO model
    ppo_agent = PPOAgent(env).to(device)
    ppo_path = os.path.join(os.getcwd(), "mvp", "params", "ppo_vector_jump.pth")
    ppo_agent.load_state_dict(torch.load(ppo_path, map_location=device))

    episodde_num = 200
    # Evaluate both models for n episodes
    fmppo_returns = evaluate_model(fmppo_agent, env, device, num_episodes=episodde_num)
    ppo_returns = evaluate_model(ppo_agent, env, device, num_episodes=episodde_num)

    # Plot the results for n episodes
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, episodde_num+1), fmppo_returns, label="FM-PPO", marker='o')
    plt.plot(range(1, episodde_num+1), ppo_returns, label="PPO", marker='o')
    plt.title("Episode Returns for FM-PPO and PPO")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.show()