import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal
import torch.nn as nn
from sofppo_constrain import Args, Agent as SFMPPOAgent, make_env
from ppo import Agent as PPOAgent
import random

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
    args = Args()
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, args.exp_name, args.gamma) for i in range(args.num_envs)]
    )

    # Load the FM-PPO model
    sfmppo_agent = SFMPPOAgent(envs).to(device)
    sfmppo_path = os.path.join(os.getcwd(), "sfm", "params", "sfmppo/sfmppo_try.pth")
    sfmppo_agent.load_state_dict(torch.load(sfmppo_path, map_location=device))

    # Load the PPO model
    ppo_agent = PPOAgent(envs).to(device)
    ppo_path = os.path.join(os.getcwd(), "sfm", "params", "ppo/ppo_no_flip_jump_intention.pth")
    ppo_agent.load_state_dict(torch.load(ppo_path, map_location=device))

    episode_num = 100
    sfmppo_returns = evaluate_model(sfmppo_agent, envs, device, num_episodes=episode_num)
    ppo_returns = evaluate_model(ppo_agent, envs, device, num_episodes=episode_num)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(sfmppo_returns)+1), sfmppo_returns, label="SFM-PPO", marker='o')
    plt.plot(range(1, len(ppo_returns)+1), ppo_returns, label="PPO", marker='o')
    # plt.title("Episode Returns for PPO & SFM-PPO On 0.8 PA/POMDP, Change of Dynamics")
    plt.title("Episode Returns for PPO & SFM-PPO On More Delayed Half-Cheetah Env")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.show()