import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sfmppo_ewc import Args, Agent as SFMPPOAgent, make_env as make_env_with_render

class EnhancedActivationVisualizer:
    def __init__(self, agent, envs, device, method='pca', fig_size=(20, 10)):
        self.agent = agent
        self.envs = envs
        self.device = device
        self.method = method
        self.fig_size = fig_size
        
        # Initialize visualization
        plt.ion()
        self.fig = plt.figure(figsize=self.fig_size)
        
        # Create subplots: env view and neural network spaces
        self.ax_env = self.fig.add_subplot(231)
        self.ax_latent = self.fig.add_subplot(232)
        self.ax_actor = self.fig.add_subplot(233)
        self.ax_critic = self.fig.add_subplot(234)
        self.ax_forward = self.fig.add_subplot(235)
        self.ax_inverse = self.fig.add_subplot(236)
        
        # Initialize PCAs for all spaces
        self.latent_pca = PCA(n_components=2)
        self.actor_pca = PCA(n_components=2)
        self.critic_pca = PCA(n_components=2)
        self.forward_pca = PCA(n_components=2)
        self.inverse_pca = PCA(n_components=2)
        
        # Initialize trajectories
        self.trajectories = {
            'latent': [],
            'actor': [],
            'critic': [],
            'forward': [],
            'inverse': []
        }
        
        self.trajectory_lines = {
            'latent': None,
            'actor': None,
            'critic': None,
            'forward': None,
            'inverse': None
        }
        
        # Get initial frame
        obs, _ = self.envs.reset()
        frame = self.envs.call('render')[0]
        self.frame_shape = frame.shape

    # def get_network_activations(self, z, action=None, next_z=None):
    #     """Extract intermediate activations from all networks"""
    #     with torch.no_grad():
    #         # Actor activation
    #         actor_hidden = self.agent.actor_mean[0](z)
            
    #         # Critic activation
    #         critic_hidden = self.agent.critic[0](z)
            
    #         # Forward model activation (if action and next_z available)
    #         if action is not None and next_z is not None:
    #             forward_input = torch.cat([z, action], dim=-1)
    #             forward_hidden = self.agent.upn.dynamics[0](forward_input)
    #         else:
    #             forward_hidden = torch.zeros_like(actor_hidden)
            
    #         # Inverse model activation (if next_z available)
    #         if next_z is not None:
    #             inverse_input = torch.cat([z, next_z], dim=-1)
    #             inverse_hidden = self.agent.upn.inverse_dynamics[0](inverse_input)
    #         else:
    #             inverse_hidden = torch.zeros_like(actor_hidden)
            
    #     return {
    #         'actor': actor_hidden,
    #         'critic': critic_hidden,
    #         'forward': forward_hidden,
    #         'inverse': inverse_hidden
    #     }

    def get_network_activations(self, z, action=None, next_z=None):
        """Extract intermediate activations from all networks, need to do padding if needed"""
        with torch.no_grad():
            # Actor activation (256-dim)
            actor_hidden = self.agent.actor_mean[0](z)
            
            # Critic activation (256-dim)
            critic_hidden = self.agent.critic[0](z)
            
            # Forward model activation (64-dim)
            if action is not None and next_z is not None:
                forward_input = torch.cat([z, action], dim=-1)
                forward_hidden = self.agent.upn.dynamics[0](forward_input)
                # Pad forward hidden to match actor dimensions
                padding_size = actor_hidden.shape[-1] - forward_hidden.shape[-1]
                if padding_size > 0:
                    forward_hidden = torch.nn.functional.pad(forward_hidden, (0, padding_size))
            else:
                forward_hidden = torch.zeros_like(actor_hidden)
            
            # Inverse model activation (64-dim)
            if next_z is not None:
                inverse_input = torch.cat([z, next_z], dim=-1)
                inverse_hidden = self.agent.upn.inverse_dynamics[0](inverse_input)
                # Pad inverse hidden to match actor dimensions
                padding_size = actor_hidden.shape[-1] - inverse_hidden.shape[-1]
                if padding_size > 0:
                    inverse_hidden = torch.nn.functional.pad(inverse_hidden, (0, padding_size))
            else:
                inverse_hidden = torch.zeros_like(actor_hidden)
            
        return {
            'actor': actor_hidden,
            'critic': critic_hidden,
            'forward': forward_hidden,
            'inverse': inverse_hidden
        }
        
    def collect_initial_representations(self, num_episodes=5):
        """Collect initial representations from all networks"""
        collections = {
            'latents': [],
            'actor': [],
            'critic': [],
            'forward': [],
            'inverse': [],
            'returns': [],
            'steps': []
        }
        
        print("Collecting initial representations...")
        for episode in range(num_episodes):
            next_obs, _ = self.envs.reset()
            next_obs = torch.Tensor(next_obs).to(self.device)
            next_done = torch.zeros(self.envs.num_envs).to(self.device)
            episode_return = torch.zeros(self.envs.num_envs).to(self.device)
            step_count = 0
            
            prev_z = None
            prev_action = None
            
            while not next_done.all():
                with torch.no_grad():
                    # Get latent representation
                    z = self.agent.upn.encoder(next_obs)
                    collections['latents'].append(z.cpu().numpy())
                    
                    # Get action
                    action, _, _, _ = self.agent.get_action_and_value(next_obs)
                    
                    # Get all network activations
                    if prev_z is not None:
                        activations = self.get_network_activations(
                            prev_z, 
                            prev_action, 
                            z
                        )
                    else:
                        activations = self.get_network_activations(z)
                    
                    # Store activations
                    for key, value in activations.items():
                        collections[key].append(value.cpu().numpy())
                    
                    # Update previous states
                    prev_z = z
                    prev_action = action
                    
                    # Step environment
                    next_obs, reward, terminations, truncations, _ = self.envs.step(action.cpu().numpy())
                    next_obs = torch.Tensor(next_obs).to(self.device)
                    next_done = torch.logical_or(torch.Tensor(terminations), torch.Tensor(truncations)).to(self.device)
                    episode_return += torch.Tensor(reward).to(self.device) * (~next_done)
                    step_count += 1

            print(f"Episode {episode + 1}/{num_episodes} completed with return: {episode_return.item()}")
            collections['returns'].append(episode_return.item())
            collections['steps'].append(step_count)
        
        return collections

    def setup_visualization(self):
        """Initialize visualization with collected data"""
        collections = self.collect_initial_representations()
        
        # Fit PCAs
        # reduced_data = {
        #     'latent': self.latent_pca.fit_transform(np.vstack(collections['latents'])),
        #     'actor': self.actor_pca.fit_transform(np.vstack(collections['actor'])),
        #     'critic': self.critic_pca.fit_transform(np.vstack(collections['critic'])),
        #     'forward': self.forward_pca.fit_transform(np.vstack(collections['forward'])),
        #     'inverse': self.inverse_pca.fit_transform(np.vstack(collections['inverse']))
        # }

        # Fit each PCA separately on the respective data

        self.latent_pca.fit(np.vstack(collections['latents']))
        self.actor_pca.fit(np.vstack(collections['actor']))
        self.critic_pca.fit(np.vstack(collections['critic']))
        self.forward_pca.fit(np.vstack(collections['forward']))  # Forward data should be 64-dim
        self.inverse_pca.fit(np.vstack(collections['inverse']))  # Inverse data should be 64-dim
        
        # Reduce dimensions for visualization
        reduced_data = {
            'latent': self.latent_pca.transform(np.vstack(collections['latents'])),
            'actor': self.actor_pca.transform(np.vstack(collections['actor'])),
            'critic': self.critic_pca.transform(np.vstack(collections['critic'])),
            'forward': self.forward_pca.transform(np.vstack(collections['forward'])),
            'inverse': self.inverse_pca.transform(np.vstack(collections['inverse']))
        }
        
        # Clear all axes
        for ax in [self.ax_env, self.ax_latent, self.ax_actor, 
                  self.ax_critic, self.ax_forward, self.ax_inverse]:
            ax.clear()
        
        # Create scatter plots for each space
        axes_map = {
            'latent': self.ax_latent,
            'actor': self.ax_actor,
            'critic': self.ax_critic,
            'forward': self.ax_forward,
            'inverse': self.ax_inverse
        }
        
        self.current_points = {}
        
        for space, ax in axes_map.items():
            # Background scatter
            scatter = ax.scatter(
                reduced_data[space][:, 0],
                reduced_data[space][:, 1],
                c=np.repeat(collections['returns'], collections['steps']),
                cmap='viridis',
                alpha=0.5
            )
            plt.colorbar(scatter, ax=ax, label='Episode Return')
            
            # Current point
            self.current_points[space] = ax.scatter(
                [], [], c='red', s=200, label='Current state'
            )
            
            # Trajectory line
            self.trajectory_lines[space], = ax.plot(
                [], [], 'r-', alpha=0.5, linewidth=1
            )
            
            # Add explained variance ratio to titles
            if space == 'latent':
                var = self.latent_pca.explained_variance_ratio_
            elif space == 'actor':
                var = self.actor_pca.explained_variance_ratio_
            elif space == 'critic':
                var = self.critic_pca.explained_variance_ratio_
            elif space == 'forward':
                var = self.forward_pca.explained_variance_ratio_
            else:  # inverse
                var = self.inverse_pca.explained_variance_ratio_
                
            ax.set_title(f'{space.capitalize()} Space\nVar: {var[0]:.2f}, {var[1]:.2f}')
            ax.legend()
        
        # Initialize environment display
        self.env_image = self.ax_env.imshow(np.zeros(self.frame_shape))
        self.ax_env.set_title('Environment')
        self.ax_env.axis('off')
        
        plt.tight_layout()
        self.fig.canvas.draw()
        print("Visualization setup completed")
    
    def update_visualization(self, obs, prev_obs=None, prev_action=None, episode_return=0):
        """Update visualizations with current state"""
        try:
            # Get current representations
            with torch.no_grad():
                z = self.agent.upn.encoder(torch.Tensor(obs).to(self.device))
                prev_z = self.agent.upn.encoder(torch.Tensor(prev_obs).to(self.device)) if prev_obs is not None else None
                
                # Get all network activations
                if prev_z is not None and prev_action is not None:
                    activations = self.get_network_activations(
                        prev_z,
                        torch.Tensor(prev_action).to(self.device),
                        z
                    )
                else:
                    activations = self.get_network_activations(z)
                
                # Transform all representations
                current_points = {
                    'latent': self.latent_pca.transform(z.cpu().numpy()),
                    'actor': self.actor_pca.transform(activations['actor'].cpu().numpy()),
                    'critic': self.critic_pca.transform(activations['critic'].cpu().numpy()),
                    'forward': self.forward_pca.transform(activations['forward'].cpu().numpy()),
                    'inverse': self.inverse_pca.transform(activations['inverse'].cpu().numpy())
                }
            
            # Update current points and trajectories
            for space, point in current_points.items():
                self.current_points[space].set_offsets(point)
                self.trajectories[space].append(point[0])
                
                if len(self.trajectories[space]) > 1:
                    trajectory = np.array(self.trajectories[space])
                    self.trajectory_lines[space].set_data(
                        trajectory[:, 0], trajectory[:, 1]
                    )
            
            # Update environment rendering
            frame = self.envs.call('render')[0]
            if frame is not None:
                self.env_image.set_array(frame)
            
            # Update title with current return
            self.ax_env.set_title(f'Environment (Return: {episode_return:.2f})')
            
            # Refresh display
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
            
        except Exception as e:
            print(f"Error in update_visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def run_episode(self):
        """Run episode with visualization"""
        next_obs, _ = self.envs.reset()
        prev_obs = None
        prev_action = None
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.envs.num_envs).to(self.device)
        episode_return = torch.zeros(self.envs.num_envs).to(self.device)
        step_count = 0
        
        print("Starting new episode...")
        for space in self.trajectories:
            self.trajectories[space] = []
        
        while not next_done.all():
            self.update_visualization(
                next_obs, 
                prev_obs,
                prev_action,
                episode_return.item()
            )
            
            with torch.no_grad():
                action, _, _, _ = self.agent.get_action_and_value(next_obs)
            
            prev_obs = next_obs.cpu().numpy()
            prev_action = action.cpu().numpy()
            
            next_obs, reward, terminations, truncations, _ = self.envs.step(prev_action)
            next_obs = torch.Tensor(next_obs).to(self.device)
            next_done = torch.logical_or(torch.Tensor(terminations), torch.Tensor(truncations)).to(self.device)
            episode_return += torch.Tensor(reward).to(self.device) * (~next_done)
            step_count += 1
            
            if step_count % 100 == 0:
                print(f"Steps: {step_count}, Current Return: {episode_return.item():.2f}")
        
        final_return = episode_return.item()
        print(f"Episode completed - Steps: {step_count}, Return: {final_return:.2f}")
        return final_return

def main():
    print("Initializing environment and agent...")
    args = Args()
    # torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    envs = gym.vector.SyncVectorEnv(
        [make_env_with_render(args.env_id, 0, True, args.exp_name, args.gamma)]
    )
    print(f"Environment created: {args.env_id}")
    
    agent = SFMPPOAgent(envs).to(device)
    print(agent)
    model_path = os.path.join(os.getcwd(), "mvp", "params", "sfmppo/sfmppo_ewc_compare.pth")
    agent.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully")
    
    visualizer = EnhancedActivationVisualizer(agent, envs, device)
    visualizer.setup_visualization()
    
    num_episodes = 5
    returns = []
    for episode in range(num_episodes):
        print(f"\nStarting Episode {episode + 1}/{num_episodes}")
        episode_return = visualizer.run_episode()
        returns.append(episode_return)
        print(f"Episode {episode + 1}/{num_episodes} completed, Return: {episode_return:.2f}")
    
    print("\nAll episodes completed")
    print(f"Average Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()