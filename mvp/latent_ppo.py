import os
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from ppo_vector import Args, Agent as PPOAgent, make_env

def extract_latent_representations(ppo_model, envs, device, num_episodes=10):
    latent_reps = []
    actions = []
    observations = []
    episode_returns = []
    episode_lengths = []
    
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
                
                action, _, _, _ = ppo_model.get_action_and_value(obs_tensor)
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
    
    return np.vstack(latent_reps), episode_returns, episode_lengths

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
    ppo_model = PPOAgent(envs).to(device)
    ppo_path = os.path.join(os.getcwd(), "mvp", "params", "ppo_vector.pth")
    ppo_model.load_state_dict(torch.load(ppo_path, map_location=device))

    # Analyze latent space
    latent_reps, reduced_reps, episode_returns, episode_steps = analyze_latent_space(ppo_model, envs, device, num_episodes=100, method='pca')