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
from env_wrappers import (JumpRewardWrapper, TargetVelocityWrapper, DelayedRewardWrapper, MultiTimescaleWrapper, 
                          NoisyObservationWrapper, MultiStepTaskWrapper, PartialObservabilityWrapper, ActionMaskingWrapper)

@dataclass
class Args:
    exp_name: str = "fmppo_halfcheetah"
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 200000
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = True
    seed: int = 1
    learning_rate: float = 3e-4
    latent_size: int = 32
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
    upn_coef: float = 1.0
    load_upn: str = "fm_vector_latent_test_rnn.pth"
    mix_coord: bool = True
    num_future_steps: int = 3

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
        
        # env = MultiStepTaskWrapper(env=env, reward_goal_steps=3)
        # env = TargetVelocityWrapper(env, target_velocity=2.0)
        # env = JumpRewardWrapper(env, jump_target_height=2.0)
        # env = PartialObservabilityWrapper(env=env, observable_ratio=0.2)
        # env = ActionMaskingWrapper(env=env, mask_prob=0.2)
        # env = DelayedRewardWrapper(env, delay_steps=20)
        # env = NoisyObservationWrapper(env, noise_scale=0.1)
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

        # MDN-RNN for forward dynamics (predicts next latent state)
        self.forward_rnn = nn.LSTM(latent_dim + action_dim, 64, batch_first=True)
        self.forward_mdn = nn.Linear(64, latent_dim * 3)  # output mean, std, mixture coeff

        # MDN-RNN for inverse dynamics (predicts action from latent states)
        self.inverse_rnn = nn.LSTM(latent_dim * 2, 64, batch_first=True)
        self.inverse_mdn = nn.Linear(64, action_dim * 3)  # output mean, std, mixture coeff

    def forward(self, state, action, next_state):
        # Encode state and next state
        z = self.encoder(state)
        z_next = self.encoder(next_state)

        # Forward dynamics with MDN-RNN (predicting next latent state)
        forward_input = torch.cat([z, action], dim=-1).unsqueeze(0)  # Adding batch dimension
        h, _ = self.forward_rnn(forward_input)
        mdn_output = self.forward_mdn(h.squeeze(0))
        z_pred = self.sample_mdn(mdn_output, latent_dim=z.size(-1)) # 取出来 32, 32, 32

        # Inverse dynamics with MDN-RNN (predicting action from latent states)
        inverse_input = torch.cat([z, z_next], dim=-1).unsqueeze(0)
        h_inv, _ = self.inverse_rnn(inverse_input)
        mdn_output_inv = self.inverse_mdn(h_inv.squeeze(0))
        action_pred = self.sample_mdn(mdn_output_inv, latent_dim=action.size(-1)) # 取出来 32, 32, 32

        # Decode latent variables to reconstructed states
        state_recon = self.decoder(z)
        next_state_recon = self.decoder(z_next)
        next_state_pred = self.decoder(z_pred)

        return z, z_next, z_pred, action_pred, mdn_output.squeeze(), mdn_output_inv.squeeze(), state_recon, next_state_recon, next_state_pred

    def sample_mdn(self, mdn_output, latent_dim):
        # Extract parameters for the mixture of Gaussians
        means = mdn_output[:, :latent_dim] # consider BATCH
        stds = torch.exp(mdn_output[:, latent_dim:2*latent_dim])
        # mixture_coeffs = F.softmax(mdn_output[:, 2*latent_dim:], dim=-1)

        # Sampling from the mixture of Gaussians
        # mixture_idx = torch.multinomial(mixture_coeffs, 1).squeeze()
        sampled_latent = means + stds * torch.randn_like(stds)
        
        return sampled_latent
    
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        state_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)
        latent_dim = 32 #args.latent_size

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
        
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(z)

    def load_upn(self, file_path):
        '''Load only the UPN model parameters from the specified file path,
        Usage for transfering previously elarned experiences to this new one.'''

        if os.path.exists(file_path):
            print(f"Loading UPN parameters from {file_path}")
            self.upn.load_state_dict(torch.load(file_path))
        else:
            print(f"No existing UPN model found at {file_path}, starting with new parameters.")

def mdn_loss(mdn_output, target, latent_dim):
    ''' Compute MDN loss as the negative log-likelihood of the target under the mixture model '''
    # Ensure the output size is correct
    assert mdn_output.size(1) == latent_dim * 3, f"MDN output size mismatch, expected {latent_dim * 3}, got {mdn_output.size(1)}"

    # Extract means, stds, and mixture coefficients from the MDN output
    means = mdn_output[:, :latent_dim]
    stds = torch.exp(mdn_output[:, latent_dim:2*latent_dim])  # stds should be positive, hence exp
    mixture_coeffs = F.softmax(mdn_output[:, 2*latent_dim:], dim=-1)  # mixture coefficients should sum to 1

    # Ensure that means and stds have the same shape as target
    assert means.shape == stds.shape == target.shape, f"Shape mismatch: means {means.shape}, stds {stds.shape}, target {target.shape}"

    # Compute log-likelihood of the target under the Gaussian mixture
    dist = Normal(means, stds)
    log_probs = dist.log_prob(target)  # calculate log-probs for each component
    log_probs = torch.logsumexp(log_probs + torch.log(mixture_coeffs), dim=-1)  # combine with mixture coeffs

    return -log_probs.mean()  # return negative log-likelihood


def compute_upn_loss(upn, state, action, next_state):
    # Forward pass through UPN
    z, z_next, z_pred, action_pred, forward_mdn_output, inverse_mdn_output, state_recon, next_state_recon, next_state_pred = upn(state, action, next_state)

    # Reconstruction losses
    recon_loss = F.mse_loss(state_recon, state) + F.mse_loss(next_state_recon, next_state)

    # Consistency losses
    consistency_loss = F.mse_loss(next_state_pred, next_state)

    # print(f"MDN forward output shape: {z_pred.shape}")

    # Forward dynamics loss
    # forward_loss_mdn = mdn_loss(forward_mdn_output, z_next, latent_dim=z.size(-1))
    forward_loss = F.mse_loss(z_pred, z_next.detach())

    # Inverse dynamics loss
    # inverse_loss_mdn = mdn_loss(inverse_mdn_output, action, latent_dim=action.size(-1))
    inverse_loss = F.mse_loss(action_pred, action)

    # Total UPN loss
    upn_loss = recon_loss + forward_loss + inverse_loss + consistency_loss

    return upn_loss

def mixed_batch(ppo_states, ppo_actions, ppo_next_states, imitation_data_path='mvp/data/imitation_data_half_cheetah.npz'):
    '''3D: sample_size, env_dim, Dof_dim, no sample, concatination direclty'''

    # Load imitation data
    imitation_data = np.load(imitation_data_path)
    imitation_states = torch.FloatTensor(imitation_data['states']).to(device)
    imitation_actions = torch.FloatTensor(imitation_data['actions']).to(device)
    imitation_next_states = torch.FloatTensor(imitation_data['next_states']).to(device)

    print(f'Mixing Imitation Data of Size: {imitation_states.shape[0]}')

    # Ensure imitation data has the same 3D shape as PPO data
    if imitation_states.dim() == 2:
        imitation_states = imitation_states.unsqueeze(1)
        imitation_actions = imitation_actions.unsqueeze(1)
        imitation_next_states = imitation_next_states.unsqueeze(1)

    # Combine PPO and imitation data
    mixed_states = torch.cat([ppo_states, imitation_states], dim=0)
    mixed_actions = torch.cat([ppo_actions, imitation_actions], dim=0)
    mixed_next_states = torch.cat([ppo_next_states, imitation_next_states], dim=0)

    # Shuffle the combined data
    shuffle_indices = torch.randperm(mixed_states.shape[0])
    mixed_states = mixed_states[shuffle_indices]
    mixed_actions = mixed_actions[shuffle_indices]
    mixed_next_states = mixed_next_states[shuffle_indices]

    print(f'Total Mixed Imitation Data of Size: {mixed_states.shape[0]}')

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
    # this is only for upn
    next_obs_all = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)

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
            next_obs_all[step] = next_obs

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
        
        if args.mix_coord:
            # mixing screw things up, isolate the problem bit by bit
            obs_imitate, actions_imitate, next_obs_imitate = mixed_batch(obs, actions, next_obs_all)
        else:
            obs_imitate, actions_imitate, next_obs_imitate = obs, actions, next_obs_all

        # Mixed batch with imitation data
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # imitate mix
        b_obs_imitate = obs_imitate.reshape((-1,) + envs.single_observation_space.shape)
        b_actions_imitate = actions_imitate.reshape((-1,) + envs.single_action_space.shape)
        b_next_obs_imitate = next_obs_imitate.reshape((-1,) + envs.single_observation_space.shape) # previous error of passing the same obs help may be due to having 2 obs in action selection
        
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

                # previously pass in obs twice, solidifies state
                upn_loss = compute_upn_loss(agent.upn, b_obs_imitate[mb_inds], b_actions_imitate[mb_inds], b_next_obs_imitate[mb_inds]) #future_states[mb_inds])

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + upn_loss * args.upn_coef

                optimizer.zero_grad()
                loss.backward()

                for name, param in agent.named_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        print(f"NaN or Inf detected in gradients of {name}")
                
                # grad_norms = []
                # for name, param in agent.named_parameters():
                #     if param.grad is not None:
                #         grad_norms.append(param.grad.norm().item())
                # print("Max grad norm:", max(grad_norms))

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

    # plt.subplot(2, 3, 2)
    # plt.plot(metrics["episodic_lengths"])
    # plt.title('Episodic Lengths')
    # plt.xlabel('Episode')
    # plt.ylabel('Length')

    plt.subplot(2, 3, 2)
    plt.plot(metrics["approx_kls"])
    plt.title('Approx KLs')
    plt.xlabel('Episode')
    plt.ylabel('Approx KLs')

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

    data_filename = f"fmppo_vector_latent_test_rnn.pth"
    data_path = os.path.join(save_dir, data_filename)

    data_filename = f"fm_vector_latent_test_rnn.pth"
    data2_path = os.path.join(save_dir, data_filename)

    print('Saved at: ', data_path)
    torch.save(agent.state_dict(), data_path)

    print('Saved at: ', data2_path)
    torch.save(agent.upn.state_dict(), data2_path)