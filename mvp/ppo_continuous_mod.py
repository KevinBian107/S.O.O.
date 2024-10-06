import os
import random
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.distributions import Normal
from collections import deque
from dataclasses import dataclass
import gym
from gym.envs.registration import register
from gym.wrappers import NormalizeObservation, TransformObservation, NormalizeReward, TransformReward

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters using Args class
@dataclass
class Args:
    exp_name: str = "ppo_pendulum"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    env_id: str = "Pendulum-v1"
    total_timesteps: int = 1e6  # Fix here, should be a full million
    learning_rate: float = 3e-4
    num_envs: int = 1
    num_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    ent_coef: float = 0.3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.01
    anneal_lr: bool = True
    clip_vloss: bool = True

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

# Register the environment
register(
    id="Pendulum-v1",
    entry_point='env_pendulum:PendulumEnv',
    max_episode_steps=200,
)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
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
            layer_init(nn.Linear(64, 64)),
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

def create_env(seed=None):
    env = gym.make("Pendulum-v1")
    if seed is not None:
        env.reset(seed=seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    return env

def plot_reward(episode_rewards, show_result=False, avg_interval=10000):
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

    if len(reward_episode) >= avg_interval:
        avg_rewards = reward_episode.unfold(0, avg_interval, avg_interval).mean(1)
        avg_rewards_repeated = avg_rewards.repeat_interleave(avg_interval)
        avg_rewards_repeated = avg_rewards_repeated[:len(reward_episode)]
        plt.plot(avg_rewards_repeated.numpy(), label=f"Average Reward Every {avg_interval} Episodes", linestyle='--', color='red')

    plt.legend()
    plt.draw()  # Update the plot
    plt.pause(0.001)

def plot_kl(kl_values, avg_interval=100, show_result=False):
    plt.figure(2)
    kl_episode = torch.tensor(kl_values, dtype=torch.float)
    kl_episode = (kl_episode - kl_episode.mean()) / (kl_episode.std() + 1e-8)

    if show_result:
        plt.title("KL Divergence Result")
    else:
        plt.clf()
        plt.title("KL Divergence Over Time")

    plt.xlabel("Episode")
    plt.ylabel("Normalized KL Divergence")
    plt.plot(kl_episode.numpy(), label="KL Divergence", color="blue")

    if len(kl_episode) >= avg_interval:
        avg_kl = kl_episode.unfold(0, avg_interval, avg_interval).mean(1)
        avg_kl_repeated = avg_kl.repeat_interleave(avg_interval)
        avg_kl_repeated = avg_kl_repeated[:len(kl_episode)]
        plt.plot(avg_kl_repeated.numpy(), label=f"Average KL Every {avg_interval} Episodes", linestyle='--', color='cyan')

    plt.legend()
    plt.draw()  # Update the plot
    plt.pause(0.001)

def main():
    args = Args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    env = create_env(seed=args.seed)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor_critic = ActorCritic(n_states, n_actions).to(device)
    optimizer = optim.Adam(actor_critic.parameters(), lr=args.learning_rate)

    memory = PPOMemory(args.batch_size)
    episode_rewards = []

    state, _ = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)

    kl_values = []
    for iteration in range(int(args.num_iterations)):

        if args.anneal_lr:
            frac = 1.0 - (iteration / args.num_iterations)
            lr_now = frac * args.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_now

        for step in range(args.num_steps):
            with torch.no_grad():
                action, log_prob, _, value = actor_critic.get_action_and_value(state)

            next_state, reward, done, _, _ = env.step(action.cpu().numpy()[0])
            memory.store_memory(state.cpu().numpy()[0], action.cpu().numpy()[0], reward, value.item(), log_prob.item())

            state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            episode_rewards.append(reward)

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
        for step in reversed(range(args.num_steps)):
            delta = memory.rewards[step] + args.gamma * values[step + 1] - values[step]
            gae = delta + args.gamma * args.gae_lambda * gae
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

        for _ in range(args.update_epochs):
            for batch in batches:
                _, new_log_probs, entropy, new_values = actor_critic.get_action_and_value(states[batch], actions[batch])
                log_ratio = new_log_probs - old_log_probs[batch]
                ratio = log_ratio.exp()

                # Policy loss
                pg_loss1 = -advantages[batch] * ratio
                pg_loss2 = -advantages[batch] * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if args.clip_vloss:
                    value_loss_clipped = (returns[batch] - new_values).pow(2)
                    value_loss_unclipped = (returns[batch] - returns[batch]).pow(2)
                    value_loss = torch.max(value_loss_clipped, value_loss_unclipped).mean()
                else:
                    value_loss = (returns[batch] - new_values).pow(2).mean()

                loss = pg_loss - args.ent_coef * entropy.mean() + args.vf_coef * value_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor_critic.parameters(), args.max_grad_norm)
                optimizer.step()

        # Log KL divergence and plot it
        with torch.no_grad():
            approx_kl = ((log_ratio - (ratio - 1)).mean()).item()
            kl_values.append(approx_kl)
        # plot_kl(kl_values)

        memory.clear_memory()

        if iteration % 10 == 0:
            print(f'Iteration {iteration * args.batch_size}')

        if iteration >= args.num_iterations:
            break

    print("Training complete")
    plot_reward(episode_rewards, show_result=True)
    plot_kl(kl_values, show_result=True)

    # Save the model
    save_dir = os.path.join(os.getcwd(), 'mvp', 'params')
    os.makedirs(save_dir, exist_ok=True)

    data_filename = f"ppo_{args.env_id}_run.pth"
    data_path = os.path.join(save_dir, data_filename)

    print('Saved at: ', data_path)
    torch.save(actor_critic.state_dict(), data_path)

    # Ensure all plots stay visible after training
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
