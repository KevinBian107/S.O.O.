import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from scipy.optimize import minimize

from config import args_sof, args_supp

# --------------------------------------FOR-----SUPP-----MODELS--------------------------------------

def load_supp_data(file_path):
    '''Need to normalize the input, only for supp model'''
    data = np.load(file_path)
    states = torch.FloatTensor(data['states']).to(args_supp.device)
    actions = torch.FloatTensor(data['actions']).to(args_supp.device)
    next_states = torch.FloatTensor(data['next_states']).to(args_supp.device)

    states = (states - states.mean()) / (states.std() + 1e-8)
    actions = (actions - actions.mean()) / (actions.std() + 1e-8)

    return states, actions, next_states

def compute_supp_upn_loss(upn, state, action, next_state):
    '''Compute loss only for supp model'''
    z, z_next, z_pred, action_pred, state_recon, next_state_recon, next_state_pred = upn(state, action, next_state)
    recon_loss = nn.MSELoss()(state_recon, state) + nn.MSELoss()(next_state_recon, next_state)
    consistency_loss = nn.MSELoss()(next_state_pred, next_state)
    forward_loss = nn.MSELoss()(z_pred, z_next.detach())
    inverse_loss = nn.MSELoss()(action_pred, action)
    total_loss = recon_loss + forward_loss + inverse_loss + consistency_loss
    
    latent_regularization = torch.mean(torch.norm(z, p=2, dim=-1)) + torch.mean(torch.norm(z_next, p=2, dim=-1))
    total_loss += 0.01 * latent_regularization

    l2_penalty = torch.mean(torch.norm(action_pred, p=2, dim=-1))
    total_loss += inverse_loss + 0.01 * l2_penalty

    return total_loss, recon_loss, forward_loss, inverse_loss, consistency_loss

def plot_supp_losses(train_losses, val_losses):
    '''plotting specifically for supp models'''
    plt.figure(figsize=(15, 10))
    loss_types = ['Total', 'Reconstruction', 'Forward', 'Inverse', 'Consistency']
    for i, loss_type in enumerate(loss_types):
        plt.subplot(2, 3, i+1)
        plt.plot([losses[i] for losses in train_losses], label='Train')
        plt.plot([losses[i] for losses in val_losses], label='Validation')
        plt.title(f'{loss_type} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    plt.tight_layout()
    plt.savefig('supervised_upn_vae.png')
    plt.show()

# --------------------------------------FOR-----SOF-----AND-----PPO-----MODELS--------------------------------------


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    '''Only on Actor Critic, instantiate layers'''
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

def compute_hidden_action_distribution(agent, state, advantage, epsilon_k, eta_k):
    """
    Compute the softened intention policy distribution (optimal action distribution) based on the current base policy and advantage values.
    This approximates the EM algorithm's expectation step, adjusting the policy softly towards higher-advantage actions.
    """
    with torch.no_grad():
        mu, logvar = agent.upn.encode(state)
        z = agent.upn.reparameterize(mu, logvar)
        action_mean, action_std = agent.actor_mean(z), agent.actor_logstd.exp()
        base_dist = Normal(action_mean, action_std)
        # eta_k = optimize_eta_k(state, advantage, base_dist, epsilon_k)
        # print(eta_k)

        # Softened intention distribution using advantage weights
        weights = (advantage.view(-1, 1) / eta_k).exp()
        safe_stddev = torch.clamp(base_dist.stddev * weights, min=1e-6)

        hidden_dist = Normal(base_dist.mean * weights, safe_stddev)
        # print(base_dist.mean.shape)
    
    return hidden_dist


def compute_lagrangian_kl_constraint(agent, state, eta_k, epsilon_k, hidden_dist):
    """Compute KL divergence between optimal "soften" intention disytribution and current base control policy distribution"""
    with torch.no_grad():
        # is this still needed?
        # action_latent_mean, action_latent_var = agent.get_transformed_action_distribution(z)
        mu, logvar = agent.upn.encode(state)
        z = agent.upn.reparameterize(mu, logvar)
        action_mean, action_std = agent.actor_mean(z), agent.actor_logstd.exp()
        ppo_dist = Normal(action_mean, torch.exp(action_std))
        kl_div = torch.distributions.kl_divergence(hidden_dist, ppo_dist).mean()
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

def compute_eta_k_loss(agent, b_advantages, epsilon_k):
    """
    Computes the eta_k loss to enforce KL constraint using advantages (A_k).
    """
    # Calculate the log-sum-exp with A_k (advantages) and the scaling by eta_k
    log_sum_exp_term = torch.logsumexp(b_advantages / agent.eta_k, dim=-1).mean()
    
    # Construct the full loss for eta_k optimization
    eta_loss = agent.eta_k * epsilon_k + agent.eta_k * log_sum_exp_term
    return eta_loss


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
    data_path = os.path.join(save_dir, args_sof.imitation_data_path)
    imitation_data = np.load(data_path)
    imitation_states = torch.FloatTensor(imitation_data['states']).to(args_sof.device)
    imitation_actions = torch.FloatTensor(imitation_data['actions']).to(args_sof.device)
    imitation_next_states = torch.FloatTensor(imitation_data['next_states']).to(args_sof.device)

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