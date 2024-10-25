import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
import os
import argparse

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
LEARNING_RATE = 2e-4
NUM_ENVS = 1
NUM_STEPS = 2048
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPOCHS = 10
CLIP_EPSILON = 0.2
ENT_COEF = 0.0
VF_COEF = 0.5
UPN_COEF = 0.5  # New hyperparameter for UPN loss
MAX_GRAD_NORM = 0.5
BATCH_SIZE = NUM_STEPS * NUM_ENVS
MINIBATCH_SIZE = BATCH_SIZE // 32
ITERATIONS = 300

ANNEAL_LR = True
CLIP_VLOSS = True
TARGET_KL = 0.01
KL_COEF = 0.1
UPN_MIX_COEF = 0.8 # higher means more ppo action

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
    '''Design Load in for just fmppo's upn for continually building representations,
    do sequential changes in gravity or task goals (including changing back gravity) to see performance'''

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
        '''
        Want to encode imitation state, should enter network, 
        change here, forward modle takes in the state of paired imitation coordinate. 
        
        Called by compute_upn_loss as upn.

        Let the forward model here learn the dynamics from imitation data and one line optimize from gradient

        Also need to have new experiences in it so it can continually learn new skills
        
        state here is one sampled state already
        '''
        z = self.encoder(state)
        z_next = self.encoder(next_state)
        z_pred = self.dynamics(torch.cat([z, action], dim=-1))
        action_pred = self.inverse_dynamics(torch.cat([z, z_next], dim=-1))
        state_recon = self.decoder(z)
        next_state_pred = self.decoder(z_pred)
        return z, z_next, z_pred, action_pred, state_recon, next_state_pred

class ActorCriticUPN(nn.Module):
    '''connected to UPN as well'''
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
        '''Main contribution of UPN, action envisioned added to selection phase,
        the data passed in here should be all state data for performing ppo update.
        
        Using this action directly converges, though not the best action
        '''

        z = self.upn.encoder(state)
        action_mean, action_std, value = self(state) # action from actor network
        probs = Normal(action_mean, action_std)

        # Predict the next state using forward dynamics model (if needed for refinement)
        z_next = self.upn.encoder(state)  # Simulate the next state from the encoder
        action_pred = self.upn.inverse_dynamics(torch.cat([z, z_next], dim=-1))  # UPN predicted action

        if action is None:
            action = probs.sample()

        # # Combine actions from PPO policy and UPN's inverse dynamics model
        # # Weight the actions 0.5 for now, do Kalman filter thing later, AlphaGo
        combined_action = UPN_MIX_COEF * action + (1 - UPN_MIX_COEF) * action_pred

        return combined_action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value
    
    def load_upn(self, file_path):
        '''Load only the UPN model parameters from the specified file path,
        Usage for transfering previously elarned experiences to this new one.'''

        if os.path.exists(file_path):
            print(f"Loading UPN parameters from {file_path}")
            self.upn.load_state_dict(torch.load(file_path))
        else:
            print(f"No existing UPN model found at {file_path}, starting with new parameters.")

def mixed_batch(ppo_states, ppo_actions, ppo_next_states, data_path='mvp/data/pendulum_ppo_imitation_run_4.npz'):
    '''Computing mixed next state for ONLY upn data input, loss, and optimization'''

    data = np.load(data_path)
    states = data['states']
    actions = data['actions']
    next_states = data['next_states']
    # rand = np.random.randint(len(states))
    # state, action, next_state = states[rand], actions[rand], next_states[rand]

    imitation_states = torch.FloatTensor(states).to(device)
    imitation_actions = torch.FloatTensor(actions).to(device)
    imitation_next_states = torch.FloatTensor(next_states).to(device)

    mixed_states = torch.cat([imitation_states, ppo_states], dim=0)
    mixed_actions = torch.cat([imitation_actions, ppo_actions], dim=0)
    mixed_next_states = torch.cat([imitation_next_states, ppo_next_states], dim=0)

    # Shuffle the mixed data
    num_samples = mixed_states.shape[0]
    indices = torch.randperm(num_samples)  # Generate random permutation of indices

    # Apply the random permutation to shuffle the mixed data
    mixed_states = mixed_states[indices]
    mixed_actions = mixed_actions[indices]
    mixed_next_states = mixed_next_states[indices]

    return mixed_states, mixed_actions, mixed_next_states


def compute_upn_loss(upn, state, action, next_state):
    '''Main contribution of UPN, forward envisioning process contributing to the loss'''

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

def plot_reward(episode_rewards, show_result=False, avg_interval=20*BATCH_SIZE):
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
    if len(reward_episode) >= avg_interval:
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

def get_args():
    '''For args parsing'''
    parser = argparse.ArgumentParser(description="Train a PPO agent with UPN")

    # Argument to specify the path to load/save UPN parameters
    parser.add_argument('--load-upn', type=str, default=None,
                        help='Path to load UPN model parameters (default: None)')
    args = parser.parse_args()
    return args

def main(args):
    env = gym.make("Pendulum-v1")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor_critic = ActorCriticUPN(n_states, n_actions).to(device)

    if args.load_upn is not None:
        # Define the path to save and load UPN weights
        model_dir = os.path.join(os.getcwd(), 'mvp', 'params')
        os.makedirs(model_dir, exist_ok=True)
        load_path = os.path.join(model_dir, args.load_upn)

        # Attempt to load UPN weights
        actor_critic.load_upn(load_path)

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

        # mixes batch for ONLY upn update purpose
        mixed_states, mixed_actions, mixed_next_states = mixed_batch(states, actions, next_states)

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
                
                # Compute KL divergence
                approx_kl = (log_ratio - (ratio - 1)).mean()

                # UPN loss
                upn_loss = compute_upn_loss(actor_critic.upn, mixed_states[batch], mixed_actions[batch], mixed_next_states[batch])

                # Total loss
                loss = pg_loss - ENT_COEF * entropy.mean() + VF_COEF * value_loss + UPN_COEF * upn_loss + KL_COEF * approx_kl

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor_critic.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                # approx_kl = (ratio - 1.0 - log_ratio).mean()
                # if TARGET_KL is not None and approx_kl > TARGET_KL:
                #     print(f"Early stopping at iteration {iteration} due to reaching target KL.")
                #     break

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
    save_dir = os.path.join(os.getcwd(),'mvp', 'params')
    os.makedirs(save_dir, exist_ok=True)

    import re
    existing_files = os.listdir(save_dir)
    run_numbers = [int(re.search(r'run_(\d+)', f).group(1)) for f in existing_files if re.search(r'run_(\d+)', f)]
    run_number = max(run_numbers) + 1 if run_numbers else 1

    data_filename = f"fmppo_ITER_{ITERATIONS}_KL_{TARGET_KL}_RUN_{run_number}.pth"
    data_path = os.path.join(save_dir, data_filename)

    data_filename = f"fm_ITER_{ITERATIONS}_KL_{TARGET_KL}_RUN_{run_number}.pth"
    data2_path = os.path.join(save_dir, data_filename)

    print('Saved at: ', data_path)
    torch.save(actor_critic.state_dict(), data_path)

    print('Saved at: ', data2_path)
    torch.save(actor_critic.upn.state_dict(), data2_path)

if __name__ == "__main__":
    # python mvp/fmppo_continuous.py --load-upn [directly give file name]
    args = get_args()
    main(args)