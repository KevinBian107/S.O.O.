import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sys
import os
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mvp.ppo import Args, Agent as PPOAgent, make_env

class PPOLatentVisualizer:
    def __init__(self, agent, envs, device, method='pca', fig_size=(15, 5)):
        self.agent = agent
        self.envs = envs
        self.device = device
        self.method = method
        self.fig_size = fig_size
        
        # Initialize visualization
        plt.ion()
        self.fig = plt.figure(figsize=self.fig_size)
        
        # Create subplots
        self.ax_env = self.fig.add_subplot(121)
        self.ax_latent = self.fig.add_subplot(122)
        
        # Initialize PCA
        self.pca = PCA(n_components=2)
        self.current_trajectory = []
        self.trajectory_line = None
        
        # Get initial frame
        obs, _ = self.envs.reset()
        frame = self.envs.call('render')[0]
        self.frame_shape = frame.shape
        
    def get_latent_representation(self, obs):
        """Extract latent representation from PPO agent's actor network"""
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            # Get the action mean as our latent representation
            latent = self.agent.actor_mean(obs_tensor)
            return latent.cpu().numpy()
    
    def collect_initial_latents(self, num_episodes=5):
        """Collect initial latent representations with proper episodic structure"""
        latents = []
        episode_returns = []
        episode_steps = []
        
        print("Collecting initial latent representations...")
        for episode in range(num_episodes):
            next_obs, _ = self.envs.reset()
            episode_return = np.zeros(self.envs.num_envs)
            step_count = 0
            done = np.zeros(self.envs.num_envs, dtype=bool)
            
            while not done.all():
                # Get latent representation
                latent = self.get_latent_representation(next_obs)
                latents.append(latent)
                
                # Get action and step environment
                with torch.no_grad():
                    action, _, _, _ = self.agent.get_action_and_value(
                        torch.FloatTensor(next_obs).to(self.device)
                    )
                next_obs, reward, terminated, truncated, _ = self.envs.step(action.cpu().numpy())
                done = terminated | truncated
                episode_return += reward * (~done)  # Only add reward if not done
                step_count += 1

            print(f"Episode {episode + 1}/{num_episodes} completed with return: {episode_return[0]:.2f}")
            episode_returns.append(episode_return[0])  # Take first env's return
            episode_steps.append(step_count)
        
        return np.vstack(latents), episode_returns, episode_steps

    def setup_visualization(self):
        """Initialize visualization with collected data"""
        initial_latents, episode_returns, episode_steps = self.collect_initial_latents()
        self.reduced_latents = self.pca.fit_transform(initial_latents)
        
        # Clear previous plots
        self.ax_env.clear()
        self.ax_latent.clear()
        
        # Plot background points with returns-based coloring
        scatter = self.ax_latent.scatter(
            self.reduced_latents[:, 0], 
            self.reduced_latents[:, 1], 
            c=np.repeat(episode_returns, episode_steps), 
            cmap='viridis', 
            alpha=0.5
        )
        plt.colorbar(scatter, label='Episode Return')
        
        # Setup current point and trajectory
        self.current_point = self.ax_latent.scatter(
            [], [], 
            c='red', 
            s=100, 
            label='Current state'
        )
        self.trajectory_line, = self.ax_latent.plot(
            [], [], 
            'r-', 
            alpha=0.5, 
            linewidth=1
        )
        
        self.ax_latent.set_title('PPO Action Space Visualization')
        self.ax_latent.legend()
        
        # Initialize environment display
        self.env_image = self.ax_env.imshow(np.zeros(self.frame_shape))
        self.ax_env.set_title('Environment')
        self.ax_env.axis('off')
        
        plt.tight_layout()
        self.fig.canvas.draw()
        print("Visualization setup completed")
    
    def update_visualization(self, obs, episode_return):
        """Update visualizations with current state"""
        try:
            # Get current latent representation
            current_latent = self.get_latent_representation(obs)
            current_latent_reduced = self.pca.transform(current_latent)
            
            # Update current point
            self.current_point.set_offsets(current_latent_reduced)
            
            # Update trajectory
            self.current_trajectory.append(current_latent_reduced[0])
            if len(self.current_trajectory) > 1:
                trajectory = np.array(self.current_trajectory)
                self.trajectory_line.set_data(trajectory[:, 0], trajectory[:, 1])
            
            # Update environment rendering
            frame = self.envs.call('render')[0]
            if frame is not None:
                self.env_image.set_array(frame)
            
            # Update title with current return (take first env's return if vectorized)
            return_value = episode_return[0] if isinstance(episode_return, (np.ndarray, list)) else episode_return
            self.ax_env.set_title(f'Environment (Return: {return_value:.2f})')
            
            # Refresh display
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
            
        except Exception as e:
            print(f"Error in update_visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def run_episode(self):
        """Run a single episode with visualization"""
        obs, _ = self.envs.reset()
        done = np.zeros(self.envs.num_envs, dtype=bool)
        episode_return = np.zeros(self.envs.num_envs)
        step_count = 0
        
        print("Starting new episode...")
        self.current_trajectory = []  # Clear trajectory at start
        
        while not done.all():
            # Update visualization
            self.update_visualization(obs, episode_return)
            
            # Get action and step environment
            with torch.no_grad():
                action, _, _, _ = self.agent.get_action_and_value(
                    torch.FloatTensor(obs).to(self.device)
                )
            
            obs, reward, terminated, truncated, _ = self.envs.step(action.cpu().numpy())
            done = terminated | truncated
            episode_return += reward * (~done)  # Only add reward if not done
            step_count += 1
            
            if step_count % 100 == 0:
                print(f"Steps: {step_count}, Current Return: {episode_return[0]:.2f}")
        
        final_return = episode_return[0]  # Take first env's return
        print(f"Episode completed - Total Steps: {step_count}, Final Return: {final_return:.2f}")
        return final_return

def main():
    print("Initializing environment and agent...")
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create environment with render mode
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, 0, True, args.exp_name, args.gamma)]
    )
    print(f"Environment created: {args.env_id}")
    
    # Load model
    agent = PPOAgent(envs).to(device)
    model_path = os.path.join(os.getcwd(), "mvp", "params", "ppo_vector.pth")
    agent.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully")
    
    # Create and run visualizer
    visualizer = PPOLatentVisualizer(agent, envs, device)
    visualizer.setup_visualization()
    
    # Run episodes
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