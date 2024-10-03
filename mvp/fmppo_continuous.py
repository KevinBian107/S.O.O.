import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import gym
from gym.envs.registration import register
from gym.wrappers import RecordVideo

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
ENT_COEF = 0.0
VF_COEF = 0.5
UPN_COEF = 0.1  # New hyperparameter for UPN loss
MAX_GRAD_NORM = 0.5
BATCH_SIZE = NUM_STEPS * NUM_ENVS
MINIBATCH_SIZE = BATCH_SIZE // 32
ITERATIONS = 1000

ANNEAL_LR = True
CLIP_VLOSS = True
TARGET_KL = 0.01

register(
    id="Pendulum-v1",
    entry_point='env_pendulum:PendulumEnv',
    max_episode_steps=200,
)

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.returns = []
        self.advantages = []
        self.next_states = []  # Added to store next states
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states), np.array(self.actions), np.array(self.log_probs), \
               np.array(self.returns), np.array(self.advantages), np.array(self.next_states), batches

    def store_memory(self, state, action, reward, value, log_prob, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.next_states.append(next_state)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.returns = []
        self.advantages = []
        self.next_states = []

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

class ActorCriticUPN(nn.Module):
    def __init__(self, n_states, n_actions, latent_dim=32):
        super(ActorCriticUPN, self).__init__()
        self.upn = UPN(n_states, n_actions, latent_dim)
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(1, n_actions))

    def forward(self, state):
        z = self.upn.encoder(state)
        value = self.critic(z)
        action_mean = self.actor(z)
        action_std = self.log_std.exp().expand_as(action_mean)
        return action_mean, action_std, value

    def get_action_and_value(self, state, action=None):
        z = self.upn.encoder(state)
        action_mean, action_std, value = self(state)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

def compute_upn_loss(upn, state, action, next_state):
    z, z_next, z_pred, action_pred, state_recon, next_state_pred = upn(state, action, next_state)
    
    # Reconstruction loss
    recon_loss = F.mse_loss(state_recon, state) + F.mse_loss(next_state_pred, next_state)
    
    # Forward model loss
    forward_loss = F.mse_loss(z_pred, z_next.detach())
    
    # Inverse model loss
    inverse_loss = F.mse_loss(action_pred, action)
    
    # Total UPN loss
    upn_loss = recon_loss + forward_loss + inverse_loss
    return upn_loss

def plot_reward(episode_rewards, show_result=False):
    plt.figure(1)
    reward_episode = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.plot(reward_episode.numpy())
    if len(reward_episode) >= 100:
        means = reward_episode.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def main():
    env = gym.make("Pendulum-v1")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor_critic = ActorCriticUPN(n_states, n_actions).to(device)
    optimizer = optim.Adam(actor_critic.parameters(), lr=LEARNING_RATE)

    memory = PPOMemory(BATCH_SIZE)
    episode_rewards = []

    state, _ = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)

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
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            memory.store_memory(state.cpu().numpy()[0], action.cpu().numpy()[0], reward, value.item(), log_prob.item(), next_state.cpu().numpy()[0])

            state = next_state
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
        for step in reversed(range(NUM_STEPS)):
            delta = memory.rewards[step] + GAMMA * values[step + 1] - values[step]
            gae = delta + GAMMA * GAE_LAMBDA * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

        memory.returns = returns
        memory.advantages = advantages

        # PPO update
        states, actions, old_log_probs, returns, advantages, next_states, batches = memory.generate_batches()
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        next_states = torch.FloatTensor(next_states).to(device)

        for _ in range(PPO_EPOCHS):
            for batch in batches:
                _, new_log_probs, entropy, new_values = actor_critic.get_action_and_value(states[batch], actions[batch])
                log_ratio = new_log_probs - old_log_probs[batch]
                ratio = log_ratio.exp()

                # Policy loss
                pg_loss1 = -advantages[batch] * ratio
                pg_loss2 = -advantages[batch] * torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if CLIP_VLOSS:
                    memory_values = torch.FloatTensor(memory.values).to(device)
                    value_pred_clipped = memory_values[batch] + torch.clamp(
                        new_values - memory_values[batch], -CLIP_EPSILON, CLIP_EPSILON
                    )
                    value_loss_clipped = (value_pred_clipped - returns[batch]) ** 2
                    value_loss_unclipped = (new_values - returns[batch]) ** 2
                    value_loss = 0.5 * torch.max(value_loss_clipped, value_loss_unclipped).mean()
                else:
                    value_loss = 0.5 * F.mse_loss(new_values, returns[batch])

                # UPN loss
                upn_loss = compute_upn_loss(actor_critic.upn, states[batch], actions[batch], next_states[batch])

                # Total loss
                loss = pg_loss - ENT_COEF * entropy.mean() + VF_COEF * value_loss + UPN_COEF * upn_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor_critic.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                approx_kl = (ratio - 1.0 - log_ratio).mean()
                if TARGET_KL is not None and approx_kl > TARGET_KL:
                    print(f"Early stopping at iteration {iteration} due to reaching target KL.")
                    break

        memory.clear_memory()

        if iteration % 10 == 0:
            print(f'Iteration {iteration}')

        if iteration >= ITERATIONS:
            break

    print("Training complete")
    plot_reward(episode_rewards, show_result=True)
    plt.ioff()
    plt.show()

    # Save the model
    torch.save(actor_critic.state_dict(), "mvp/params/pendulum_fmppo_continuous.pth")

    # Uncomment the following lines if you want to render and save a video
    # video_env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True)
    # state, _ = video_env.reset()
    # state = torch.FloatTensor(state).unsqueeze(0).to(device)
    # done = False
    # while not done:
    #     with torch.no_grad():
    #         action, _, _, _ = actor_critic.get_action_and_value(state)
    #     state, _, done, _, _ = video_env.step(action.cpu().numpy()[0])
    #     state = torch.FloatTensor(state).unsqueeze(0).to(device)
    # video_env.close()

if __name__ == "__main__":
    main()