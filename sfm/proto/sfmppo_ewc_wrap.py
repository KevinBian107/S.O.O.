import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import defaultdict
import gymnasium as gym
from typing import Callable, Optional
from torch.utils.data import DataLoader, TensorDataset
from sfmppo import Agent, Args, make_env, compute_upn_loss
from env_wrappers import (JumpRewardWrapper, TargetVelocityWrapper, DelayedRewardWrapper, MultiTimescaleWrapper, 
                          NoisyObservationWrapper, MultiStepTaskWrapper, PartialObservabilityWrapper, ActionMaskingWrapper,
                          NonLinearDynamicsWrapper, DelayedHalfCheetahEnv)

class EWCArgs(Args):
    """Extended arguments for EWC training"""
    def __init__(self, base_args):
        for attr_name, attr_value in vars(base_args).items():
            setattr(self, attr_name, attr_value)
        
        self.ewc_lambda = 5000.0  # EWC importance factor
        self.fisher_sample_size = 200  # Samples for Fisher computation
        self.consolidation_step = 1000  # Steps before weight consolidation
        self.importance_threshold = 0.1  # Threshold for important weights
        self.base_model_path = None
        self.task_sequence_dir = "change_task_ewc"

def make_ewc_envs(args: EWCArgs, task_wrapper: Optional[Callable] = None) -> gym.vector.SyncVectorEnv:
    """
    Creates a vectorized environment setup identical to SFMPPO but with optional task wrapper.
    
    Args:
        args: EWCArgs instance containing environment parameters
        task_wrapper: Optional wrapper function to apply specific task modifications
    
    Returns:
        gym.vector.SyncVectorEnv: Vectorized environment matching SFMPPO setup
    """
    def make_single_env(env_id: str, idx: int, capture_video: bool, 
                       run_name: str, gamma: float) -> Callable:
        def thunk() -> gym.Env:
            # Create base environment exactly as in SFMPPO
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.FlattenObservation(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            
            if capture_video and idx == 0:
                env = gym.wrappers.RecordVideo(
                    env,
                    f"videos/{run_name}",
                    step_trigger=lambda step: step % 1000 == 0
                )
                
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
            
            # Apply task-specific wrapper if provided
            if task_wrapper is not None:
                env = task_wrapper(env)
                
            return env
            
        return thunk

    # Create vectorized environment exactly as in SFMPPO
    envs = gym.vector.SyncVectorEnv(
        [make_single_env(args.env_id, i, False, args.exp_name, args.gamma) 
         for i in range(args.num_envs)]
    )
    return envs

class FisherInformation:
    """Handles Fisher Information computation and storage"""
    def __init__(self, agent, device):
        self.agent = agent
        self.device = device
        self.fisher_matrices = {}
        self.parameter_means = {}
        
    def compute_fisher_matrix(self, data_loader, num_samples):
        fisher_diagonals = {}
        for name, param in self.agent.named_parameters():
            fisher_diagonals[name] = torch.zeros_like(param)
            
        self.agent.eval()
        samples_processed = 0
        
        for states, actions, _, _, _, _ in data_loader:
            if samples_processed >= num_samples:
                break
                
            # Compute log probabilities for actions
            self.agent.zero_grad()
            _, log_probs, _, _ = self.agent.get_action_and_value(states, actions)
            log_prob_mean = log_probs.mean()
            
            # Compute gradients
            log_prob_mean.backward()
            
            # Accumulate squared gradients
            for name, param in self.agent.named_parameters():
                if param.grad is not None:
                    fisher_diagonals[name] += param.grad.data.pow(2)
                    
            samples_processed += states.size(0)
            
        # Normalize
        for name in fisher_diagonals:
            fisher_diagonals[name] /= samples_processed
            
        return fisher_diagonals
    
    def update_task_fisher(self, task_id, data_loader, num_samples):
        """Update Fisher Information for a specific task"""
        self.fisher_matrices[task_id] = self.compute_fisher_matrix(
            data_loader, num_samples
        )
        self.parameter_means[task_id] = {
            name: param.data.clone()
            for name, param in self.agent.named_parameters()
        }

class EWCSFMPPOTrainer:
    """Hierarchical trainer that extends SFMPPO with EWC capabilities"""
    def __init__(self, args: EWCArgs, base_env_id: str, device: str):
        self.args = args
        self.device = device
        self.base_env = make_ewc_envs(self.args)
        self.agent = Agent(self.base_env).to(device)
        
        # Load base model if specified
        if args.base_model_path:
            self.load_base_model(args.base_model_path)
        
        self.fisher_info = FisherInformation(self.agent, device)
        
        # Optimizers
        self.setup_optimizers()
        
        # Metrics tracking
        self.metrics = defaultdict(list)
    
    def load_base_model(self, model_path):
        """Load and verify base model"""
        if os.path.exists(model_path):
            self.agent.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded base model from {model_path}")
        else:
            raise FileNotFoundError(f"Base model not found at {model_path}")
    
    def create_data_loader(self, episode_data, batch_size=64, shuffle=True):
        """
        Creates a DataLoader from episode data for efficient batch processing.
        
        Args:
            episode_data (dict): Dictionary containing tensors of states, actions, etc.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data.
            
        Returns:
            DataLoader: PyTorch DataLoader for batching the episode data.
        """
        # Convert dictionary of tensors to list of tensors in specific order
        dataset = TensorDataset(
            episode_data['states'],
            episode_data['actions'],
            episode_data['rewards'],
            episode_data['log_probs'],
            episode_data['next_states'],
            episode_data['dones']
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False
        )

    def collect_episode_data(self, env):
        """Collects data for an episode from the environment for training."""
        states, actions, rewards, log_probs, next_states, dones = [], [], [], [], [], []
        obs, _ = env.reset()
        
        episode_length = 0
        max_episode_length = 10
        
        done = False
        while not done and episode_length < max_episode_length:
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            with torch.no_grad():
                action, log_prob, _, _ = self.agent.get_action_and_value(obs_tensor)

            action = action.cpu().numpy()
            next_obs, reward, termination, truncation, info = env.step(action)
            done = termination or truncation

            # Store the collected data
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob.cpu().numpy())
            next_states.append(next_obs)
            dones.append(done)

            obs = next_obs
            episode_length += 1

        return {
            'states': torch.FloatTensor(np.array(states)).to(self.device),
            'actions': torch.FloatTensor(np.array(actions)).to(self.device),
            'rewards': torch.FloatTensor(np.array(rewards)).to(self.device),
            'log_probs': torch.FloatTensor(np.array(log_probs)).to(self.device),
            'next_states': torch.FloatTensor(np.array(next_states)).to(self.device),
            'dones': torch.FloatTensor(np.array(dones)).to(self.device)
        }
        
    def setup_optimizers(self):
        """Initialize optimizers with separate parameter groups"""
        self.ppo_optimizer = torch.optim.Adam([
            {'params': self.agent.actor_mean.parameters()},
            {'params': self.agent.actor_logstd},
            {'params': self.agent.critic.parameters()}
        ], lr=self.args.ppo_learning_rate)
        
        self.upn_optimizer = torch.optim.Adam(
            self.agent.upn.parameters(),
            lr=self.args.upn_learning_rate
        )
            
    def calculate_gae(self, rewards, values, dones):
        """Calculates advantages and returns using GAE."""
        advantages = torch.zeros_like(rewards).to(self.device)
        returns = torch.zeros_like(rewards).to(self.device)
        lastgaelam = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = 0  # Assuming the episode ends here, next value is 0
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]

            delta = rewards[t] + self.args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam

        returns = advantages + values
        return advantages, returns

    def compute_ewc_loss(self, current_task_id):
        """Compute EWC loss across all previous tasks"""
        ewc_loss = torch.tensor(0., device=self.device)
        
        for task_id, fisher_matrix in self.fisher_info.fisher_matrices.items():
            if task_id != current_task_id:
                for name, param in self.agent.named_parameters():
                    mean = self.fisher_info.parameter_means[task_id][name]
                    fisher = fisher_matrix[name]
                    
                    # Compute quadratic penalty
                    ewc_loss += (fisher * (param - mean).pow(2)).sum()
                    
        return self.args.ewc_lambda * ewc_loss
    
    def train_step(self, env_wrapper, current_task_id, episode_data):
        """Single training step with EWC regularization"""
        states = episode_data['states']
        actions = episode_data['actions']
        rewards = episode_data['rewards']
        log_probs = episode_data['log_probs']
        next_states = episode_data['next_states']
        dones = episode_data['dones']
        values = self.agent.get_value(states).detach()

        # Calculate advantages using GAE
        advantages, returns = self.calculate_gae(rewards, values, dones)

        # Standard SFMPPO losses
        _, new_log_probs, entropy, new_value = self.agent.get_action_and_value(states, actions)

        # PPO losses
        ratio = torch.exp(new_log_probs - log_probs)
        policy_loss1 = -advantages * ratio
        policy_loss2 = -advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
        policy_loss = torch.max(policy_loss1, policy_loss2).mean()  # Ensure this is a scalar
        value_loss = 0.5 * ((new_value - returns) ** 2).mean()  # Ensure this is a scalar

        # UPN losses
        recon_loss, forward_loss, inverse_loss, consistency_loss = compute_upn_loss(
            self.agent.upn, states, actions, next_states
        )
        upn_loss = (recon_loss + forward_loss + inverse_loss + consistency_loss) * self.args.upn_coef

        # EWC loss
        ewc_loss = self.compute_ewc_loss(current_task_id)

        # Total loss
        total_loss = (
            policy_loss +
            value_loss * self.args.vf_coef +
            upn_loss +
            ewc_loss -
            entropy.mean() * self.args.ent_coef  # Ensure entropy is a scalar
        )

        # Optimization step
        self.ppo_optimizer.zero_grad()
        self.upn_optimizer.zero_grad()
        total_loss.backward()

        # Clip gradients
        nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)

        self.ppo_optimizer.step()
        self.upn_optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'upn_loss': upn_loss.item(),
            'ewc_loss': ewc_loss.item(),
            'entropy': entropy.mean().item()
        }

    def train_task(self, task_wrapper, task_id, num_episodes):
        """Train on a specific task while preserving previous knowledge"""
        env = make_ewc_envs(self.args, task_wrapper)
        episode_metrics = []
        
        for episode in range(num_episodes):
            episode_data = self.collect_episode_data(env)
            step_metrics = self.train_step(task_wrapper, task_id, episode_data)
            episode_metrics.append(step_metrics)
            
            # Update Fisher Information periodically
            if episode % self.args.consolidation_step == 0:
                data_loader = self.create_data_loader(
                    episode_data,
                    batch_size=min(64, len(episode_data['states']))
                )
                self.fisher_info.update_task_fisher(
                    task_id,
                    data_loader,
                    self.args.fisher_sample_size
                )
                
            # Save checkpoint
            if episode % 100 == 0:
                self.save_checkpoint(task_id, episode)
                
        return episode_metrics
        
    def train_sequence(self, task_wrappers):
        """Train on a sequence of tasks"""
        os.makedirs(self.args.task_sequence_dir, exist_ok=True)
        
        for task_id, task_wrapper in enumerate(task_wrappers):
            print(f"\nTraining Task {task_id}")
            metrics = self.train_task(task_wrapper, task_id, self.args.total_timesteps)
            
            # Log and save metrics
            self.log_metrics(metrics, task_id)
            self.save_checkpoint(task_id, final=True)
            
        self.plot_training_curves()
        
    def save_checkpoint(self, task_id, episode=None, final=False):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.agent.state_dict(),
            'fisher_matrices': self.fisher_info.fisher_matrices,
            'parameter_means': self.fisher_info.parameter_means,
            'metrics': self.metrics
        }
        
        filename = f"task_{task_id}_final.pt" if final else \
                  f"task_{task_id}_episode_{episode}.pt"
        path = os.path.join(self.args.task_sequence_dir, filename)
        torch.save(checkpoint, path)
        
    def plot_training_curves(self):
        """Plot training metrics across all tasks"""
        plt.figure(figsize=(15, 10))
        
        # Plot returns
        plt.subplot(2, 2, 1)
        for task_id in range(len(self.metrics['returns'])):
            plt.plot(self.metrics['returns'][task_id], 
                    label=f'Task {task_id}')
        plt.title('Episode Returns')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.legend()
        
        # Plot losses
        plt.subplot(2, 2, 2)
        for task_id in range(len(self.metrics['ewc_losses'])):
            plt.plot(self.metrics['ewc_losses'][task_id], 
                    label=f'Task {task_id}')
        plt.title('EWC Losses')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('ewc_training_curves.png')
        plt.close()

if __name__ == "__main__":
    base_args = Args()
    save_dir = os.path.join(os.getcwd(), 'mvp', 'params')
    data_path = os.path.join(save_dir, base_args.save_sfmppo)
    
    ewc_args = EWCArgs(base_args)
    ewc_args.base_model_path = data_path
    
    task_wrappers = [
        lambda env: JumpRewardWrapper(env, jump_target_height=1.0),
        lambda env: TargetVelocityWrapper(env, target_velocity=2.0),
        lambda env: DelayedRewardWrapper(env, delay_steps=10)
    ]
    
    trainer = EWCSFMPPOTrainer(
        args=ewc_args,
        base_env_id="HalfCheetah-v4",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    trainer.train_sequence(task_wrappers)