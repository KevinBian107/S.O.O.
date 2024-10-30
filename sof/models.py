import os
import numpy as np
from config import args_sof, args_ppo
import torch
import torch.nn as nn
from torch.distributions import Normal

from utils import *

class UPN(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim):
        super(UPN, self).__init__()
        
        # Encoder outputs mean and log variance for VAE
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, args_sof.upn_hidden_layer),
            nn.ReLU(),
            nn.Linear(args_sof.upn_hidden_layer, args_sof.upn_hidden_layer),
            nn.ReLU(),
        )
        self.enc_mean = nn.Linear(args_sof.upn_hidden_layer, latent_dim)
        self.enc_logvar = nn.Linear(args_sof.upn_hidden_layer, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, args_sof.upn_hidden_layer),
            nn.ReLU(),
            nn.Linear(args_sof.upn_hidden_layer, args_sof.upn_hidden_layer),
            nn.ReLU(),
            nn.Linear(args_sof.upn_hidden_layer, state_dim)
        )
        
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, args_sof.upn_hidden_layer),
            nn.ReLU(),
            nn.Linear(args_sof.upn_hidden_layer, args_sof.upn_hidden_layer),
            nn.ReLU(),
            nn.Linear(args_sof.upn_hidden_layer, latent_dim)
        )
        
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(latent_dim * 2, args_sof.upn_hidden_layer),
            nn.ReLU(),
            nn.Linear(args_sof.upn_hidden_layer, args_sof.upn_hidden_layer),
            nn.ReLU(),
            nn.Linear(args_sof.upn_hidden_layer, action_dim)
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
    
class Agent_sof(nn.Module):
    def __init__(self, envs):
        super().__init__()
        state_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)
        latent_dim = args_sof.latent_size

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
            layer_init(nn.Linear(latent_dim, args_sof.ppo_hidden_layer)),
            nn.Tanh(),
            layer_init(nn.Linear(args_sof.ppo_hidden_layer, args_sof.ppo_hidden_layer)),
            nn.Tanh(),
            layer_init(nn.Linear(args_sof.ppo_hidden_layer, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_dim, args_sof.ppo_hidden_layer)),
            nn.Tanh(),
            layer_init(nn.Linear(args_sof.ppo_hidden_layer, args_sof.ppo_hidden_layer)),
            nn.Tanh(),
            layer_init(nn.Linear(args_sof.ppo_hidden_layer, action_dim), std=0.01),
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


class Agent_ppo(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), args_ppo.ppo_hidden_layer)),
            nn.Tanh(),
            layer_init(nn.Linear(args_ppo.ppo_hidden_layer, args_ppo.ppo_hidden_layer)),
            nn.Tanh(),
            layer_init(nn.Linear(args_ppo.ppo_hidden_layer, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), args_ppo.ppo_hidden_layer)),
            nn.Tanh(),
            layer_init(nn.Linear(args_ppo.ppo_hidden_layer, args_ppo.ppo_hidden_layer)),
            nn.Tanh(),
            layer_init(nn.Linear(args_ppo.ppo_hidden_layer, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)