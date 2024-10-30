import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal
import torch.nn as nn
import random

from environments import make_env
from config import args_test
from models import Agent_ppo as PPOAgent, Agent_sof as SOFAgent

def evaluate_model(agent, envs, device, num_episodes=100):
    '''Evaluate models, models all imported from taring files, customize wrappers,
    perfectly consistent with both ppo and fmppo, previous environment is inconsistent, very delicate.'''
    returns = []
    for episode in range(num_episodes):
        next_obs, _ = envs.reset()
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(envs.num_envs).to(device)
        episode_returns = torch.zeros(envs.num_envs).to(device)

        while not next_done.all():
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(next_obs)
                next_obs, reward, terminations, truncations, _ = envs.step(action.cpu().numpy())
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.logical_or(torch.Tensor(terminations), torch.Tensor(truncations)).to(device)
                episode_returns += torch.Tensor(reward).to(device) * (~next_done)
                print(f'Episode: {episode}/{num_episodes} With Rewards: {episode_returns}')

        returns.extend(episode_returns.cpu().numpy())

    return returns

if __name__ == "__main__":
    random.seed(args_test.seed)
    np.random.seed(args_test.seed)
    torch.manual_seed(args_test.seed)
    torch.backends.cudnn.deterministic = args_test.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args_test.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args_test.env_id, i, args_test.capture_video, args_test.exp_name, args_test.gamma) for i in range(args_test.num_envs)]
    )

    # Load the SOF-PPO model
    sfmppo_agent = SOFAgent(envs).to(device)
    sfmppo_path = os.path.join(os.getcwd(), "sof", "params", "sofppo", args_test.sof_path)
    sfmppo_agent.load_state_dict(torch.load(sfmppo_path, map_location=device))

    # Load the PPO model
    ppo_agent = PPOAgent(envs).to(device)
    ppo_path = os.path.join(os.getcwd(), "sof", "params", "ppo", args_test.ppo_path)
    ppo_agent.load_state_dict(torch.load(ppo_path, map_location=device))

    episode_num = args_test.test_episode_num
    sfmppo_returns = evaluate_model(sfmppo_agent, envs, device, num_episodes=episode_num)
    ppo_returns = evaluate_model(ppo_agent, envs, device, num_episodes=episode_num)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(sfmppo_returns)+1), sfmppo_returns, label="SOF-PPO", marker='o')
    plt.plot(range(1, len(ppo_returns)+1), ppo_returns, label="PPO", marker='o')
    plt.title("Episode Returns for Intention Constrain Models Evaluated in Intention Environment")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.show()