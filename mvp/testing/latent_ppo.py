import os
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from ppo_vector import Args, Agent as PPOAgent, make_env

class ModifiedPPOAgent(PPOAgent):
    def get_state_representation(self, x):
        # Extract the output of the second-to-last layer of the actor network
        for i, layer in enumerate(self.actor_mean):
            x = layer(x)
            if i == len(self.actor_mean) - 2:  # Second-to-last layer
                return x
        return x

def extract_latent_representations(ppo_model, envs, device, num_episodes=10):
    '''latent representations in ppo is the action directly, next action is random'''
    latent_reps = []
    actions = []
    observations = []
    episode_returns = []
    episode_lengths = []
    state_reps = []
    
    for episode in range(num_episodes):
        obs, _ = envs.reset()
        done = False
        episode_return = 0
        episode_length = 0
        
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action_mean = ppo_model.actor_mean(obs_tensor)
                latent_reps.append(action_mean.cpu().numpy().reshape(-1))  # Flatten latent representations

                state_rep = ppo_model.get_state_representation(obs_tensor)
                state_reps.append(state_rep.cpu().numpy().reshape(-1))
                
                action, _, _, _ = ppo_model.get_action_and_value(obs_tensor)
                # action = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
                action = action.cpu().numpy().squeeze()  # Remove the batch dimension
                actions.append(action)
                observations.append(obs)
            
            obs, reward, terminated, truncated, _ = envs.step(action)
            episode_return += reward
            episode_length += 1
            done = terminated or truncated
        
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}/{num_episodes}, Return: {episode_return}, Length: {episode_length}")
    
    return np.vstack(state_reps), episode_returns, episode_lengths

def reduce_dimensionality(latent_reps, method='pca', n_components=2):
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
    else:
        raise ValueError("Method must be either 'pca' or 'tsne'")
    
    return reducer.fit_transform(latent_reps)

def visualize_latent_space(reduced_reps, episode_returns, episode_steps, method):
    plt.figure(figsize=(15, 5))
    
    # Plot latent space colored by episode return
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(reduced_reps[:, 0], reduced_reps[:, 1], c=np.repeat(episode_returns, episode_steps), cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Episode Return')
    plt.title(f'Latent Space Visualization using {method.upper()} (colored by return)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    
    # Plot latent space colored by step within episode
    plt.subplot(1, 2, 2)
    step_colors = np.concatenate([np.arange(steps) for steps in episode_steps])
    scatter = plt.scatter(reduced_reps[:, 0], reduced_reps[:, 1], c=step_colors, cmap='coolwarm', alpha=0.5)
    plt.colorbar(scatter, label='Step within Episode')
    plt.title(f'Latent Space Visualization using {method.upper()} (colored by step)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    
    plt.tight_layout()
    plt.show()

def analyze_latent_space(agent, envs, device, num_episodes=10, method='pca'):
    # Extract latent representations
    latent_reps, episode_returns, episode_steps = extract_latent_representations(agent, envs, device, num_episodes)
    
    # Reduce dimensionality
    reduced_reps = reduce_dimensionality(latent_reps, method)
    
    # Visualize
    visualize_latent_space(reduced_reps, episode_returns, episode_steps, method)

    return latent_reps, reduced_reps, episode_returns, episode_steps

if __name__ == "__main__":
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, args.exp_name, args.gamma) for i in range(args.num_envs)]
    )

    # Create and load the PPO model
    ppo_model = ModifiedPPOAgent(envs).to(device)
    ppo_path = os.path.join(os.getcwd(), "mvp", "params", "ppo_vector.pth")
    ppo_model.load_state_dict(torch.load(ppo_path, map_location=device))

    # Analyze latent space
    latent_reps, reduced_reps, episode_returns, episode_steps = analyze_latent_space(ppo_model, envs, device, num_episodes=100, method='pca')

    # Additional analysis: Correlation between latent dimensions and episode returns
    episode_latents = np.array([np.mean(latent_reps[sum(episode_steps[:i]):sum(episode_steps[:i+1])], axis=0) for i in range(len(episode_returns))])
    correlations = np.corrcoef(episode_latents.T, episode_returns)[:-1, -1]
    top_corr_dims = np.argsort(np.abs(correlations))[-5:][::-1]
    
    print("\nTop 5 latent dimensions correlated with episode returns:")
    for dim in top_corr_dims:
        print(f"Dimension {dim}: Correlation {correlations[dim]:.3f}")

    # Visualize these top dimensions
    plt.figure(figsize=(15, 10))
    for i, dim in enumerate(top_corr_dims):
        plt.subplot(2, 3, i+1)
        plt.scatter(episode_latents[:, dim], episode_returns)
        plt.title(f"Dimension {dim}")
        plt.xlabel("Mean Latent Value")
        plt.ylabel("Episode Return")
    plt.tight_layout()
    plt.show()