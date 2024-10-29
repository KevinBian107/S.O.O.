import os
import time
import random
import re
from dataclasses import dataclass
from collections import deque

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from scipy.optimize import minimize
import matplotlib.pyplot as plt
from gymnasium.experimental.wrappers.rendering import RecordVideoV0 as RecordVideo
from env_wrappers import (JumpRewardWrapper, TargetVelocityWrapper, DelayedRewardWrapper, MultiTimescaleWrapper, 
                          NoisyObservationWrapper, MultiStepTaskWrapper, PartialObservabilityWrapper, ActionMaskingWrapper,
                          NonLinearDynamicsWrapper, DelayedHalfCheetahEnv)

@dataclass
class Args:
    exp_name: str = "sofppo_halfcheetah"
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 1000000
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = True
    seed: int = 1
    ppo_learning_rate: float = 1e-5
    upn_learning_rate: float = 8e-6 # lower learning rate
    latent_size: int = 100
    upn_hidden_layer: int = 64
    ppo_hidden_layer: int = 256
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
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    upn_coef: float = 0.8
    kl_coef: float = 0.3

    # exactly how far we want distribution to be
    # what's good for suboptimal
    epsilon_k: float = 0.01
    eta_k: float = 0.01

    # when constrain_weights = 0, no constrain on MOMPO constrain
    constrain_weights: float = 0.5

    # this helps greatly
    mix_coord: bool = False
    
    # Data need to match up, this data may be problematic
    load_upn: str = "supp/supervised_vae_jump.pth"
    load_sfmppo: str = "sfmppo/sfmppo_stable.pth" # can still use this becuase only load in PPO

    imitation_data_path: str= None #"imitation_data_ppo_new.npz"
    save_sfm: str = "sfm/sfm_try.pth"
    save_sfmppo: str = "sfmppo/sfmppo_try.pth"

    # to be set at runtime
    batch_size: int = 0 
    minibatch_size: int = 0
    iterations: int = 0

args = Args()

def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx==0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = RecordVideo(env, f"videos/{run_name}")
            # fixed it by reading Stack Overfloat
        else:
            env = gym.make(env_id)
        
        # env = TargetVelocityWrapper(env, target_velocity=2.0)
        # env = JumpRewardWrapper(env, jump_target_height=2.0)
        # env = PartialObservabilityWrapper(env=env, observable_ratio=0.2)
        # env = ActionMaskingWrapper(env=env, mask_prob=0.2)
        # env = DelayedRewardWrapper(env, delay_steps=20)
        # env = NonLinearDynamicsWrapper(env, dynamic_change_threshold=50)
        # env = NoisyObservationWrapper(env, noise_scale=0.1)
        # env = DelayedHalfCheetahEnv(env=env, proprio_delay=1, force_delay=3)
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

def freeze_base_controller(agent):
    """Freeze all parameters in the base controller (actor and critic)"""
    for param in agent.actor_mean.parameters():
        param.requires_grad = False
    for param in agent.critic.parameters():
        param.requires_grad = False
    agent.actor_logstd.requires_grad = False

def freeze_intention(agent):
    """Freeze all parameters in the base controller (actor and critic)"""
    for param in agent.upn.parameters():
        param.requires_grad = False
    
def unfreeze_base_controller(agent):
    """Unfreeze all parameters in the base controller if needed"""
    for param in agent.actor_mean.parameters():
        param.requires_grad = True
    for param in agent.critic.parameters():
        param.requires_grad = True
    agent.actor_logstd.requires_grad = True

class UPN(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim):
        super(UPN, self).__init__()
        
        # Encoder outputs mean and log variance for VAE
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, args.upn_hidden_layer),
            nn.ReLU(),
            nn.Linear(args.upn_hidden_layer, args.upn_hidden_layer),
            nn.ReLU(),
        )
        self.enc_mean = nn.Linear(args.upn_hidden_layer, latent_dim)
        self.enc_logvar = nn.Linear(args.upn_hidden_layer, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, args.upn_hidden_layer),
            nn.ReLU(),
            nn.Linear(args.upn_hidden_layer, args.upn_hidden_layer),
            nn.ReLU(),
            nn.Linear(args.upn_hidden_layer, state_dim)
        )
        
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, args.upn_hidden_layer),
            nn.ReLU(),
            nn.Linear(args.upn_hidden_layer, args.upn_hidden_layer),
            nn.ReLU(),
            nn.Linear(args.upn_hidden_layer, latent_dim)
        )
        
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(latent_dim * 2, args.upn_hidden_layer),
            nn.ReLU(),
            nn.Linear(args.upn_hidden_layer, args.upn_hidden_layer),
            nn.ReLU(),
            nn.Linear(args.upn_hidden_layer, action_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        h = self.encoder(x)
        return self.enc_mean(h), self.enc_logvar(h)

    def forward(self, state, action, next_state):
        # Encode current state
        mu, logvar = self.encode(state)
        z = self.reparameterize(mu, logvar)
        
        # Encode next state
        mu_next, logvar_next = self.encode(next_state)
        z_next = self.reparameterize(mu_next, logvar_next)
        
        # Forward dynamics in latent space
        z_pred = self.dynamics(torch.cat([z, action], dim=-1))
        
        # Inverse dynamics
        action_pred = self.inverse_dynamics(torch.cat([z, z_next], dim=-1))
        
        # Decode states
        state_recon = self.decoder(z)
        next_state_pred = self.decoder(z_pred)
        next_state_recon = self.decoder(z_next)
        
        return z, z_next, z_pred, action_pred, state_recon, next_state_recon, next_state_pred, \
               mu, logvar, mu_next, logvar_next
    
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        state_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)
        latent_dim = args.latent_size

        self.upn = UPN(state_dim, action_dim, latent_dim)
        self.action_mean_to_latent = nn.Sequential(
            nn.Linear(action_dim, latent_dim), 
            nn.Tanh(), 
            nn.Linear(latent_dim, latent_dim)
        )
        self.action_var_to_latent = nn.Sequential(
            nn.Linear(action_dim, latent_dim), 
            nn.Tanh(), 
            nn.Linear(latent_dim, latent_dim)
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_dim, args.ppo_hidden_layer)),
            nn.Tanh(),
            layer_init(nn.Linear(args.ppo_hidden_layer, args.ppo_hidden_layer)),
            nn.Tanh(),
            layer_init(nn.Linear(args.ppo_hidden_layer, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_dim, args.ppo_hidden_layer)),
            nn.Tanh(),
            layer_init(nn.Linear(args.ppo_hidden_layer, args.ppo_hidden_layer)),
            nn.Tanh(),
            layer_init(nn.Linear(args.ppo_hidden_layer, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        # Use the VAE encoder to get the mean and log variance
        mu, logvar = self.upn.encode(x)
        # Reparameterize to sample z from the distribution
        z = self.upn.reparameterize(mu, logvar)
        return self.critic(z)

    def get_action_and_value(self, x, action=None):
        mu, logvar = self.upn.encode(x)
        z = self.upn.reparameterize(mu, logvar)
        action_mean = self.actor_mean(z)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(z)
    
    def get_transformed_action_distribution(self, z):
        """ Map action space to latent space dimension, both action mean and action logstd"""
        action_mean = self.actor_mean(z)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_latent_mean = self.action_mean_to_latent(action_mean)
        action_latent_var = self.action_var_to_latent(action_logstd)
        return action_latent_mean, action_latent_var
    
    def load_upn(self, file_path):
        '''Load only the UPN model parameters from the specified file path,
        Usage for transfering previously elarned experiences to this new one.'''

        if os.path.exists(file_path):
            print(f"Loading UPN parameters from {file_path}")
            self.upn.load_state_dict(torch.load(file_path))
        else:
            print(f"No existing UPN model found at {file_path}, starting with new parameters.")
    
    def load_ppo(self, file_path):
        '''Load only the PPO model parameters (actor and critic only) from the specified file path.'''
        if os.path.exists(file_path):
            print(f"Loading PPO parameters from {file_path}")
            checkpoint = torch.load(file_path)
            
            # Selectively load the PPO-related parameters
            ppo_state_dict = {k: v for k, v in checkpoint.items() if 
                              'actor_mean' in k or 'critic' in k or 'actor_logstd' in k}
            
            self.load_state_dict(ppo_state_dict, strict=False)
        else:
            print(f"No existing PPO model found at {file_path}, starting with new parameters.")

def compute_intention_action_distribution(agent, state, advantage, epsilon_k):
    """
    Compute the softened intention policy distribution (optimal action distribution) based on the current base policy and advantage values.
    This approximates the EM algorithm's expectation step, adjusting the policy softly towards higher-advantage actions.
    """
    with torch.no_grad():
        action_mean, action_std = agent.actor_mean(state), agent.actor_logstd.exp()
        base_dist = Normal(action_mean, action_std)
        eta_k = optimize_eta_k(state, advantage, base_dist, epsilon_k)

        # Softened intention distribution using advantage weights
        weights = (advantage / eta_k).exp()

        intention_dist = Normal(base_dist.mean * weights, base_dist.stddev * weights)
    
    return intention_dist, eta_k


def compute_lagrangian_kl_constraint(agent, state, eta_k, epsilon_k, intention_dist):
    """Compute KL divergence between optimal "soften" intention disytribution and current base control policy distribution"""
    with torch.no_grad():
        mu, logvar = agent.upn.encode(state)
        z = agent.upn.reparameterize(mu, logvar)

        # is this still needed?
        action_latent_mean, action_latent_var = agent.get_transformed_action_distribution(z)
        ppo_dist = Normal(action_latent_mean, torch.exp(action_latent_var))
        kl_div = torch.distributions.kl_divergence(intention_dist, ppo_dist).mean()
        constraint_violation = F.relu(kl_div - epsilon_k)
    
    return eta_k * constraint_violation


def compute_upn_loss(upn, state, action, next_state):
    '''Compute sololy UPN losses'''
    z, z_next, z_pred, action_pred, state_recon, next_state_recon, next_state_pred, \
        mu, logvar, mu_next, logvar_next = upn(state, action, next_state)
    
    recon_loss = F.mse_loss(state_recon, state) + F.mse_loss(next_state_recon, next_state)
    consistency_loss = F.mse_loss(next_state_pred, next_state)
    forward_loss = F.mse_loss(z_pred, z_next.detach())
    inverse_loss = F.mse_loss(action_pred, action)

    return recon_loss, forward_loss, inverse_loss, consistency_loss

def eta_k_objective(eta, states, advantages, old_policy, epsilon_k):
    """
    Objective function for optimizing eta_k.
    """
    eta = eta[0]  # extract scalar from array input by scipy
    log_expectation_sum = 0

    with torch.no_grad():
        for i, state in enumerate(states):
            # Sample actions from the old policy distribution
            action_mean, action_std = old_policy.mean[i], old_policy.stddev[i]
            action_dist = Normal(action_mean, action_std)
            sampled_actions = action_dist.sample((10,))  # Sample 10 actions per state

            # Compute the exponentiated advantage divided by eta
            exp_adv_scaled = (advantages[i] / eta).exp()
            weighted_adv = exp_adv_scaled * action_dist.log_prob(sampled_actions).exp()
            log_expectation = torch.log(weighted_adv.mean())
            log_expectation_sum += log_expectation.item()
        
    # Compute the full objective
    objective_value = eta * epsilon_k + eta * (log_expectation_sum / len(states))
    return objective_value

def optimize_eta_k(states, advantages, old_policy, epsilon_k):
    """
    Optimize eta_k using scipy's minimize function.
    """
    # Initial guess for eta
    eta_initial = 1.0
    result = minimize(eta_k_objective, [eta_initial], args=(states, advantages, old_policy, epsilon_k),
                      bounds=[(1e-3, None)], method="L-BFGS-B")  # Ensure eta is positive
    return result.x[0]  # Optimized eta_k

def mixed_batch(ppo_states, ppo_actions, ppo_next_states):
    '''3D: sample_size, env_dim, Dof_dim, no sample, concatination direclty'''

    # Load imitation data
    save_dir = os.path.join(os.getcwd(), 'sfm', 'data')
    os.makedirs(save_dir, exist_ok=True)
    data_path = os.path.join(save_dir, args.imitation_data_path)
    imitation_data = np.load(data_path)
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

    # freeze_base_controller(agent)
    
    if args.load_sfmppo is not None:
        save_dir = os.path.join(os.getcwd(), 'sfm', 'params')
        data_path = os.path.join(save_dir, args.load_sfmppo)
        if os.path.exists(data_path):
            print(f"Loading sfmppo model from {data_path}")
            agent.load_ppo(data_path)  # Use the new method to load only PPO parameters
        else:
            print(f"Model file not found at {data_path}. Starting training from scratch.")

    if args.load_upn is not None:
        # if args.load_sfmppo is not None:
        #     print('Loading Full model, cannot load sfm core')
        # else:
            # Define the path to save and load UPN weights
            print('loaded params for supervised forward model')
            model_dir = os.path.join(os.getcwd(), 'sfm', 'params')
            os.makedirs(model_dir, exist_ok=True)
            load_path = os.path.join(model_dir, args.load_upn)
            # Attempt to load UPN weights
            agent.load_upn(load_path)

    # Optimizer for PPO (actor and critic)
    ppo_optimizer = optim.Adam([
    {'params': agent.actor_mean.parameters()},
    {'params': agent.actor_logstd},
    {'params': agent.critic.parameters()}], lr=args.ppo_learning_rate, eps=1e-5)

    # Optimizer for UPN
    upn_optimizer = optim.Adam(agent.upn.parameters(), lr=args.upn_learning_rate, eps=1e-5)

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
        "upn_losses": [],
        "recon_losses":[],
        "forward_losses":[],
        "inverse_losses":[],
        "consist_losses":[]
    }

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.iterations
            lrnow = frac * args.ppo_learning_rate
            ppo_optimizer.param_groups[0]["lr"] = lrnow

        metrics["learning_rates"].append(ppo_optimizer.param_groups[0]["lr"])

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
                
                # if args.target_kl is not None and approx_kl > args.target_kl:
                #     print(f"Early stopping at iteration {iteration} due to reaching target KL.")
                #     break

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

                
                # Entropy loss
                entropy_loss = entropy.mean()

                # Lagrangian Objective (Adjusted with KL Intention Distribution Constraint)
                intention_dist, eta_k = compute_intention_action_distribution(agent,
                                                                              b_obs_imitate[mb_inds],
                                                                              b_advantages[mb_inds],
                                                                              args.epsilon_k
                                                                              )
                kl_constraint_penalty = compute_lagrangian_kl_constraint(agent,
                                                                         b_obs_imitate[mb_inds],
                                                                         eta_k,
                                                                         args.epsilon_k,
                                                                         intention_dist
                                                                         )
                recon_loss, forward_loss, inverse_loss, consistency_loss, kl_loss = compute_upn_loss(agent.upn,
                                                                                                     b_obs_imitate[mb_inds],
                                                                                                     b_actions_imitate[mb_inds],
                                                                                                     b_next_obs_imitate[mb_inds]
                                                                                                     )
                ppo_loss = (pg_loss -
                            args.ent_coef * entropy_loss +
                            v_loss * args.vf_coef +
                            approx_kl * args.kl_coef +
                            kl_constraint_penalty * args.constrain_weights
                            )
                # Previously not on in sfmppo
                upn_loss = args.upn_coef * (recon_loss +
                                            forward_loss +
                                            inverse_loss +
                                            consistency_loss +
                                            kl_loss * args.constrain_weights
                                            )
                
                # PPO backward pass and optimization
                ppo_optimizer.zero_grad()
                ppo_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(agent.actor_mean.parameters()) + 
                    [agent.actor_logstd] + 
                    list(agent.critic.parameters()), 
                    args.max_grad_norm
                )
                ppo_optimizer.step()

                # UPN backward pass and optimization
                upn_optimizer.zero_grad()
                upn_loss.backward()
                nn.utils.clip_grad_norm_(agent.upn.parameters(), args.max_grad_norm)
                upn_optimizer.step()

                for name, param in agent.named_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        print(f"NaN or Inf detected in gradients of {name}")
                
                # grad_norms = []
                # for name, param in agent.named_parameters():
                #     if param.grad is not None:
                #         grad_norms.append(param.grad.norm().item())
                # print("Max grad norm:", max(grad_norms))

                # nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                # # optimizer.step()

        # Logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        metrics["value_losses"].append(v_loss.item())
        metrics["policy_losses"].append(pg_loss.item())
        metrics['upn_losses'].append(upn_loss.item())
        metrics['forward_losses'].append(forward_loss.item())
        metrics['inverse_losses'].append(inverse_loss.item())
        metrics["recon_losses"].append(recon_loss.item())
        metrics["consist_losses"].append(consistency_loss.item())
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

    avg_interval = 50
    # Ensure that metrics["episodic_returns"] is a 1D list or array
    episodic_returns = np.array(metrics["episodic_returns"]).flatten()

    # Now apply np.convolve to calculate the rolling average
    if len(episodic_returns) >= avg_interval:
        avg_returns = np.convolve(episodic_returns, np.ones(avg_interval) / avg_interval, mode='valid')
        plt.plot(range(avg_interval - 1, len(episodic_returns)), avg_returns, label=f"{avg_interval}-Episode Average", color="orange")

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
    plt.plot(metrics["upn_losses"], label='UPN Loss')
    plt.plot(metrics["forward_losses"], label='Forward Loss')
    plt.plot(metrics["inverse_losses"], label='Inverse Loss')
    plt.plot(metrics["recon_losses"], label='Reconstruction Loss')
    plt.plot(metrics["consist_losses"], label='Consistency Loss')
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
    plt.savefig('sofppo_vae_constrained.png')
    plt.show()

    # Save the model
    save_dir = os.path.join(os.getcwd(), 'sfm', 'params')
    os.makedirs(save_dir, exist_ok=True)

    data1_path = os.path.join(save_dir, args.save_sfmppo)
    data2_path = os.path.join(save_dir, args.save_sfm)

    print('Saved at: ', data1_path)
    torch.save(agent.state_dict(), data1_path)

    print('Saved at: ', data2_path)
    torch.save(agent.upn.state_dict(), data2_path)