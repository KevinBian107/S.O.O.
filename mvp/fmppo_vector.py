import os
import time
import random
import re
from dataclasses import dataclass
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
from gymnasium.experimental.wrappers.rendering import RecordVideoV0 as RecordVideo

@dataclass
class Args:
    exp_name: str = "fmppo_inverted_pendulum"
    env_id: str = "InvertedPendulum-v4"
    total_timesteps: int = 100000
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = True
    seed: int = 1
    learning_rate: float = 3e-4
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 10
    num_minibatches: int = 32
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    upn_coef: float = 0.5 # planning helps
    upn_mix_coef: float = 0.9 # higher = more ppo action, action making should still be ppo
    load_upn: str = "fm_vector.pth"

    # kl_coef: float = 0.1
    # target_kl: float = None

    # to be set at runtime
    batch_size: int = 0 
    minibatch_size: int = 0
    iterations: int = 0

def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx==0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = RecordVideo(env, f"videos/{run_name}")
            # fixed it by reading Stack Overfloat
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    '''Only on Actor Critic'''
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

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
class Agent(nn.Module):
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

    def get_value(self, x):
        z = self.upn.encoder(x)
        return self.critic(z)

    def get_action_and_value(self, x, action=None):
        z = self.upn.encoder(x)
        action_mean = self.actor_mean(z)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        z_next = self.upn.encoder(x)
        action_pred = self.upn.inverse_dynamics(torch.cat([z, z_next], dim=-1))

        if action is None:
            action = probs.sample()

        combined_action = args.upn_mix_coef * action + (1 - args.upn_mix_coef) * action_pred
        return combined_action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(z)
    
    def load_upn(self, file_path):
        '''Load only the UPN model parameters from the specified file path,
        Usage for transfering previously elarned experiences to this new one.'''

        if os.path.exists(file_path):
            print(f"Loading UPN parameters from {file_path}")
            self.upn.load_state_dict(torch.load(file_path))
        else:
            print(f"No existing UPN model found at {file_path}, starting with new parameters.")

def compute_upn_loss(upn, state, action, next_state):
    z, z_next, z_pred, action_pred, state_recon, next_state_pred = upn(state, action, next_state)
    recon_loss = F.mse_loss(state_recon, state) + F.mse_loss(next_state_pred, next_state)
    forward_loss = F.mse_loss(z_pred, z_next.detach())
    inverse_loss = F.mse_loss(action_pred, action)
    upn_loss = recon_loss + forward_loss + inverse_loss
    return upn_loss

def mixed_batch(ppo_states, ppo_actions, ppo_next_states, data_path='mvp/data/imitation_data.npz'):
    # Load imitation data
    data = np.load(data_path)
    states = data['states']
    actions = data['actions']
    next_states = data['next_states']

    # Convert imitation data to tensors
    imitation_states = torch.FloatTensor(states).to(device)
    imitation_actions = torch.FloatTensor(actions).to(device)
    imitation_next_states = torch.FloatTensor(next_states).to(device)

    # Reshape imitation data similar to PPO data
    imitation_states = imitation_states.reshape((-1,) + envs.single_observation_space.shape)
    imitation_actions = imitation_actions.reshape((-1,) + envs.single_action_space.shape)
    imitation_next_states = imitation_next_states.reshape((-1,) + envs.single_observation_space.shape)

    # Print tensor shapes for debugging
    # print(f"Shape of PPO states: {ppo_states.shape}")
    # print(f"Shape of Imitation states (after reshaping): {imitation_states.shape}")
    # print(f"Shape of PPO actions: {ppo_actions.shape}")
    # print(f"Shape of Imitation actions: {imitation_actions.shape}")

    # Concatenate PPO and imitation data along the batch dimension (dim=0)
    mixed_states = torch.cat([imitation_states, ppo_states], dim=0)
    mixed_actions = torch.cat([imitation_actions, ppo_actions], dim=0)
    mixed_next_states = torch.cat([imitation_next_states, ppo_next_states], dim=0)

    # Shuffle the mixed batch
    num_samples = mixed_states.shape[0]
    indices = torch.randperm(num_samples)

    # Shuffle the concatenated data
    mixed_states = mixed_states[indices]
    mixed_actions = mixed_actions[indices]
    mixed_next_states = mixed_next_states[indices]

    return mixed_states, mixed_actions, mixed_next_states


def plot_metrics(metrics, show_result=False):
    plt.figure(figsize=(12, 8))
    plt.clf()
    plt.title("Training..." if not show_result else "Result")
    plt.xlabel("Iteration")
    plt.ylabel("Metrics")

    for key, values in metrics.items():
        plt.plot(values, label=key)

    plt.legend()
    plt.pause(0.001)

if __name__ == "__main__":
    args = Args()
    args.batch_size = args.num_steps * args.num_envs
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.iterations = args.total_timesteps // args.batch_size

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, args.exp_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)

    if args.load_upn is not None:
        # Define the path to save and load UPN weights
        print('loaded params for forward model')
        model_dir = os.path.join(os.getcwd(), 'mvp', 'params')
        os.makedirs(model_dir, exist_ok=True)
        load_path = os.path.join(model_dir, args.load_upn)

        # Attempt to load UPN weights
        agent.load_upn(load_path)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Logging setup
    global_step = 0
    start_time = time.time()
    metrics = {
        "episodic_returns": [],
        "episodic_lengths": [],
        "learning_rates": [],
        "value_losses": [],
        "policy_losses": [],
        "entropies": [],
        "approx_kls": [],
        "clipfracs": [],
        "explained_variances": [],
        "upn_losses": []
    }

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        metrics["learning_rates"].append(optimizer.param_groups[0]["lr"])

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        metrics["episodic_returns"].append(info["episode"]["r"])
                        metrics["episodic_lengths"].append(info["episode"]["l"])

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Mixed batch with imitation data
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # mixing screw things up, isolate the problem bit by bit
        # b_obs, b_actions, b_next_states = mixed_batch(b_obs, b_actions, b_obs)

        b_inds = np.arange(args.batch_size)
        clipfracs_batch = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs_batch += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                upn_loss = compute_upn_loss(agent.upn, b_obs[mb_inds], b_actions[mb_inds], b_obs[mb_inds])

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + upn_loss * args.upn_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # Logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        metrics["value_losses"].append(v_loss.item())
        metrics["policy_losses"].append(pg_loss.item())
        metrics["entropies"].append(entropy_loss.item())
        metrics["approx_kls"].append(approx_kl.item())
        metrics["clipfracs"].append(np.mean(clipfracs_batch))
        metrics["explained_variances"].append(explained_var)

        sps = int(global_step / (time.time() - start_time))
        print(f"SPS: {sps}")

    envs.close()

    # Plotting results
    plt.figure(figsize=(20, 10))

    plt.subplot(2, 3, 1)
    plt.plot(metrics["episodic_returns"])
    plt.title('Episodic Returns')
    plt.xlabel('Episode')
    plt.ylabel('Return')

    plt.subplot(2, 3, 2)
    plt.plot(metrics["episodic_lengths"])
    plt.title('Episodic Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Length')

    plt.subplot(2, 3, 3)
    plt.plot(metrics["learning_rates"])
    plt.title('Learning Rate')
    plt.xlabel('Iteration')
    plt.ylabel('LR')

    plt.subplot(2, 3, 4)
    plt.plot(metrics["value_losses"], label='Value Loss')
    plt.plot(metrics["policy_losses"], label='Policy Loss')
    plt.title('Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(metrics["entropies"])
    plt.title('Entropy')
    plt.xlabel('Iteration')
    plt.ylabel('Entropy')

    plt.subplot(2, 3, 6)
    plt.plot(metrics["explained_variances"])
    plt.title('Explained Variance')
    plt.xlabel('Iteration')
    plt.ylabel('Variance')

    plt.tight_layout()
    plt.savefig('fmppo_results.png')
    plt.show()

    # Save the model
    save_dir = os.path.join(os.getcwd(), 'mvp', 'params')
    os.makedirs(save_dir, exist_ok=True)

    data_filename = f"fmppo_vector.pth"
    data_path = os.path.join(save_dir, data_filename)

    data_filename = f"fm_vector.pth"
    data2_path = os.path.join(save_dir, data_filename)

    print('Saved at: ', data_path)
    torch.save(agent.state_dict(), data_path)

    print('Saved at: ', data2_path)
    torch.save(agent.upn.state_dict(), data2_path)