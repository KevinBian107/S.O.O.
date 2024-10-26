import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import gym
from gym.envs.registration import register
from gym.wrappers import RecordVideo
from gym.wrappers import NormalizeObservation, TransformObservation, NormalizeReward, TransformReward


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# Hyperparameters
LEARNING_RATE = 3e-4
NUM_ENVS = 1
NUM_STEPS = 2048
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPOCHS = 10
CLIP_EPSILON = 0.2
ENT_COEF = 0.3
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
BATCH_SIZE = NUM_STEPS * NUM_ENVS
MINIBATCH_SIZE = BATCH_SIZE // 16
TOTAL_STEPS = 1e6
ITERATIONS = TOTAL_STEPS // BATCH_SIZE
ANNEAL_LR = True
CLIP_VLOSS = True
TARGET_KL = 0.01 # Target KL divergence for early stopping, lower value could prevent exploration
# KL_COEF = 0.3

register(
    id="Pendulum-v1",
    entry_point='env_pendulum:PendulumEnv',
    max_episode_steps=200,
)

def create_env(seed=None):
    '''Env generate function'''
    env = gym.make("Pendulum-v1")
    if seed is not None:
        env.reset(seed=seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    env = gym.wrappers.TransformReward(env, lambda reward: reward / 16.2736044) # normalize rewards
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = gym.wrappers.ClipAction(env)
    # env = gym.wrappers.NormalizeObservation(env)
    # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    return env

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    '''Add orthogonal weight initialization for the layers for networks'''

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.returns = []
        self.advantages = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states), np.array(self.actions), np.array(self.log_probs), \
               np.array(self.returns), np.array(self.advantages), batches

    def store_memory(self, state, action, reward, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.returns = []
        self.advantages = []

class ActorCritic(nn.Module):
    def __init__(self, n_states, n_actions):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(n_states, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, n_actions)),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_states, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1))
        )
        self.log_std = nn.Parameter(torch.zeros(1, n_actions))

    def forward(self, state):
        value = self.critic(state)
        action_mean = self.actor(state)
        action_std = self.log_std.exp().expand_as(action_mean)
        return action_mean, action_std, value

    def get_action_and_value(self, state, action=None):
        action_mean, action_std, value = self(state)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

def plot_reward(episode_rewards, show_result=False, avg_interval=20*BATCH_SIZE):
    '''avg_interval need to consider BATCHED size'''

    plt.figure(1)
    reward_episode = torch.tensor(episode_rewards, dtype=torch.float)
    
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")

    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.plot(reward_episode.numpy(), label="Episode Reward")

    # Plot the moving average for every 50 episodes
    if len(reward_episode) >= avg_interval:
        # Calculate average reward for every `avg_interval` episodes
        avg_rewards = reward_episode.unfold(0, avg_interval, avg_interval).mean(1)
        avg_rewards_repeated = avg_rewards.repeat_interleave(avg_interval)
        
        # In case the number of episodes is not perfectly divisible by avg_interval, trim the result
        avg_rewards_repeated = avg_rewards_repeated[:len(reward_episode)]
        
        plt.plot(avg_rewards_repeated.numpy(), label=f"Average Reward Every {avg_interval} Episodes", linestyle='--', color='red')

    plt.legend()
    plt.pause(0.001)

    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def plot_reward_real_time(episode_rewards, show_result=False, avg_interval=20*BATCH_SIZE):
    """Plot the reward achieved"""

    plt.figure(1)
    reward_episode = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("rwards")
    # plt.plot(reward_episode.numpy())
    # Take 100 episode averages and plot them too
    if len(reward_episode) >= avg_interval:
        means = reward_episode.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def normalize_tensor(tensor):
    '''Normalize a tensor by subtracting the mean and dividing by the standard deviation'''
    mean = tensor.mean()
    std = tensor.std() + 1e-8  # Avoid division by zero
    return (tensor - mean) / std

def plot_kl(kl_values, avg_interval=20*BATCH_SIZE, show_result=False):
    '''Plot normalized KL divergence, less than reward, reward collected at each episode, kl collected whe policy updates.'''

    plt.figure(2)
    kl_episode = torch.tensor(kl_values, dtype=torch.float)

    # Normalize the KL values
    kl_episode = normalize_tensor(kl_episode)

    if show_result:
        plt.title("KL Divergence Result")
    else:
        plt.clf()
        plt.title("KL Divergence Over Time")

    plt.xlabel("Episode")
    plt.ylabel("Normalized KL Divergence")

    # Dynamically adjust avg_interval if it's too large
    min_len = len(kl_episode)
    avg_interval = min(avg_interval, min_len // 2)  # Adjust if avg_interval is too large

    # Plot the KL values
    plt.plot(kl_episode.numpy(), label="KL Divergence", color="blue")

    # Plot the moving average of the KL values
    if len(kl_episode) >= avg_interval:
        avg_kl = kl_episode.unfold(0, avg_interval, avg_interval).mean(1)
        avg_kl_repeated = avg_kl.repeat_interleave(avg_interval)
        avg_kl_repeated = avg_kl_repeated[:len(kl_episode)]
        
        plt.plot(avg_kl_repeated.numpy(), label=f"Average KL Every {avg_interval} Episodes", linestyle='--', color='cyan')

    plt.legend()
    plt.pause(0.001)

    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def main():
    env = create_env(seed=42)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor_critic = ActorCritic(n_states, n_actions).to(device)
    optimizer = optim.Adam(actor_critic.parameters(), lr=LEARNING_RATE)

    memory = PPOMemory(BATCH_SIZE)
    episode_rewards = []

    state, _ = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)

    kl_values = []
    for iteration in count(1):

        if ANNEAL_LR:
            frac = 1.0 - (iteration / ITERATIONS)
            lr_now = frac * LEARNING_RATE
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_now

        for step in range(NUM_STEPS):
            with torch.no_grad():
                action, log_prob, _, value = actor_critic.get_action_and_value(state)

            next_state, reward, done, _, _ = env.step(action.cpu().numpy()[0])
            memory.store_memory(state.cpu().numpy()[0], action.cpu().numpy()[0], reward, value.item(), log_prob.item())

            state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            episode_rewards.append(reward)

            # plotting significantly reduces speed
            # plot_reward_real_time(episode_rewards, show_result=True)

            if done:
                state, _ = env.reset()
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                plot_reward(episode_rewards)
                episode_rewards = []

        # Compute returns and advantages
        returns = []
        advantages = []
        values = memory.values + [actor_critic.get_action_and_value(state)[3].item()]
        gae = 0
        for step in reversed(range(NUM_STEPS)):
            delta = memory.rewards[step] + GAMMA * values[step + 1] - values[step]
            gae = delta + GAMMA * GAE_LAMBDA * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

        memory.returns = returns
        memory.advantages = advantages

        # PPO update
        states, actions, old_log_probs, returns, advantages, batches = memory.generate_batches()
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(PPO_EPOCHS):
            for batch in batches:
                _, new_log_probs, entropy, new_values = actor_critic.get_action_and_value(states[batch], actions[batch])
                log_ratio = new_log_probs - old_log_probs[batch]
                ratio = (new_log_probs - old_log_probs[batch]).exp()

                # Policy loss
                pg_loss1 = -advantages[batch] * ratio
                pg_loss2 = -advantages[batch] * torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                # value_loss = F.mse_loss(new_values.squeeze(), returns[batch])
                if CLIP_VLOSS:
                    # Convert memory.values to a tensor if it's not already
                    memory_values = torch.FloatTensor(memory.values).to(device)

                    # Clipped value prediction
                    value_pred_clipped = memory_values[batch] + torch.clamp(
                        new_values - memory_values[batch], -CLIP_EPSILON, CLIP_EPSILON
                    )

                    # Calculate both clipped and unclipped value losses
                    value_loss_clipped = (value_pred_clipped - returns[batch]) ** 2
                    value_loss_unclipped = (new_values - returns[batch]) ** 2

                    # Take the maximum of the two losses (as per PPO clipping)
                    value_loss = 0.5 * torch.max(value_loss_clipped, value_loss_unclipped).mean()
                else:
                    value_loss = 0.5 * F.mse_loss(new_values, returns[batch])
                
                # Compute KL divergence
                approx_kl = (log_ratio - (ratio - 1)).mean()
                kl_values.append(approx_kl.item())

                # Total loss
                loss = pg_loss - ENT_COEF * entropy.mean() + VF_COEF * value_loss #+ KL_COEF * approx_kl

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor_critic.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                # KL here is not penalty, but rather break out directly for too big of a change
                if TARGET_KL is not None and approx_kl > TARGET_KL:
                    print(f"Early stopping at iteration {iteration + 1} due to reaching target KL.")
                    break

        memory.clear_memory()

        if iteration % 10 == 0:
            print(f'Iteration {iteration * BATCH_SIZE}')

        if iteration >= ITERATIONS:  # Adjust this number based on desired training duration
            break

    print("Training complete")
    plot_reward(episode_rewards, show_result=True)
    plot_kl(kl_values, show_result=True)

    # Save the model
    save_dir = os.path.join(os.getcwd(),'mvp', 'params')
    os.makedirs(save_dir, exist_ok=True)

    import re
    existing_files = os.listdir(save_dir)
    run_numbers = [int(re.search(r'run_(\d+)', f).group(1)) for f in existing_files if re.search(r'run_(\d+)', f)]
    run_number = max(run_numbers) + 1 if run_numbers else 1

    data_filename = f"ppo_ITER_{ITERATIONS}_KL_{TARGET_KL}_RUN_{run_number}.pth"
    data_path = os.path.join(save_dir, data_filename)

    print('Saved at: ', data_path)
    torch.save(actor_critic.state_dict(), data_path)

if __name__ == "__main__":
    main()