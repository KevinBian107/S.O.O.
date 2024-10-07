import os
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal
import torch.nn as nn
from fmppo_vector import Args
from env_wrappers import JumpRewardWrapper, TargetVelocityWrapper
from fmppo_vector import Agent as FMPPOAgent
from ppo_vector import Agent as PPOAgent
import random

import os
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal
import torch.nn as nn
from fmppo_vector import Args, Agent as FMPPOAgent, make_env
from ppo_vector import Agent as PPOAgent
import random

def evaluate_model(agent, envs, device, num_episodes=100):
    '''Evaluate models, models all imported from taring files, customize wrappers,
    perfectly consistent with both ppo and fmppo, previous environment is inconsistent, very delicate.'''
    returns = []
    for episode in range(num_episodes):
        print(f'Episode: {episode}')
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
    # env = gym.vector.SyncVectorEnv([lambda: TargetVelocityWrapper(gym.make(env_id), target_velocity=1.0)])
    # env = gym.vector.SyncVectorEnv([lambda: JumpRewardWrapper(gym.make(env_id), jump_target_height=1.0)])

    # Load the FM-PPO model
    fmppo_agent = FMPPOAgent(envs).to(device)
    fmppo_path = os.path.join(os.getcwd(), "mvp", "params", "fmppo_vector_jump.pth")
    fmppo_agent.load_state_dict(torch.load(fmppo_path, map_location=device))

    # Load the PPO model
    ppo_agent = PPOAgent(envs).to(device)
    ppo_path = os.path.join(os.getcwd(), "mvp", "params", "ppo_vector_jump.pth")
    ppo_agent.load_state_dict(torch.load(ppo_path, map_location=device))

    episode_num = 100
    fmppo_returns = evaluate_model(fmppo_agent, envs, device, num_episodes=episode_num)
    ppo_returns = evaluate_model(ppo_agent, envs, device, num_episodes=episode_num)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(fmppo_returns)+1), fmppo_returns, label="FM-PPO", marker='o')
    plt.plot(range(1, len(ppo_returns)+1), ppo_returns, label="PPO", marker='o')
    plt.title("Episode Returns for FM-PPO and PPO")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.show()


