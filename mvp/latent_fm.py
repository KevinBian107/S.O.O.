import os
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from fmppo_vector import Args, UPN, make_env

def extract_latent_representations(upn_model, envs, device, num_episodes=10):
    latent_reps = []
    observations = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, _ = envs.reset()
        done = np.zeros(envs.num_envs, dtype=bool)
        episode_length = 0
        
        while not done.all():
            obs_tensor = torch.FloatTensor(obs).to(device)
            with torch.no_grad():
                latent = upn_model.encoder(obs_tensor)
                latent_reps.append(latent.cpu().numpy())
                observations.append(obs)
            
            action = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            obs, _, terminated, truncated, _ = envs.step(action)
            done = np.logical_or(terminated, truncated)
            episode_length += 1
        
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}/{num_episodes}, Length: {episode_length}")
    
    return np.vstack(latent_reps), np.vstack(observations), episode_lengths

def reduce_dimensionality(latent_reps, method='pca', n_components=2):
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
    else:
        raise ValueError("Method must be either 'pca' or 'tsne'")
    
    return reducer.fit_transform(latent_reps)

def visualize_latent_space(reduced_reps, observations, episode_lengths, method):
    plt.figure(figsize=(15, 5))

    # Plot latent space colored by episode
    plt.subplot(1, 2, 1)
    episode_colors = np.concatenate([np.full(length * envs.num_envs, i) for i, length in enumerate(episode_lengths)])
    episode_colors = episode_colors[:len(reduced_reps)]  # Ensure correct length
    scatter = plt.scatter(reduced_reps[:, 0], reduced_reps[:, 1], c=episode_colors, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Episode')
    plt.title(f'Latent Space Visualization using {method.upper()} (colored by episode)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # Plot latent space colored by a specific observation dimension (e.g., position)
    plt.subplot(1, 2, 2)
    obs_dim = 0  # Choose an observation dimension to visualize
    observation_colors = observations[:, obs_dim]
    scatter = plt.scatter(reduced_reps[:, 0], reduced_reps[:, 1], c=observation_colors, cmap='coolwarm', alpha=0.5)
    plt.colorbar(scatter, label=f'Observation Dim {obs_dim}')
    plt.title(f'Latent Space Visualization using {method.upper()} (colored by obs dim {obs_dim})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    plt.tight_layout()
    plt.show()

def analyze_latent_space(upn_model, envs, device, num_episodes=10, method='pca'):
    latent_reps, observations, episode_lengths = extract_latent_representations(upn_model, envs, device, num_episodes)
    reduced_reps = reduce_dimensionality(latent_reps, method)
    visualize_latent_space(reduced_reps, observations, episode_lengths, method)
    return latent_reps, reduced_reps, observations, episode_lengths

if __name__ == "__main__":
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, args.exp_name, args.gamma) for i in range(args.num_envs)]
    )

    # Create and load only the UPN model
    state_dim = np.array(envs.single_observation_space.shape).prod()
    action_dim = np.prod(envs.single_action_space.shape)
    latent_dim = 32  # Adjust if your UPN uses a different latent dimension
    upn_model = UPN(state_dim, action_dim, latent_dim).to(device)

    # Load only the UPN part of the model
    # fmppo_path = os.path.join(os.getcwd(), "mvp", "params", "fmppo_vector.pth")
    # fmppo_state_dict = torch.load(fmppo_path, map_location=device)
    # upn_state_dict = {k.replace('upn.', ''): v for k, v in fmppo_state_dict.items() if k.startswith('upn.')}
    # upn_model.load_state_dict(upn_state_dict)

    fm_path = os.path.join(os.getcwd(), "mvp", "params", "fm_vector.pth")
    fm_state_dict = torch.load(fm_path, map_location=device)
    upn_model.load_state_dict(fm_state_dict)

    # Analyze latent space
    latent_reps, reduced_reps, observations, episode_lengths = analyze_latent_space(upn_model, envs, device, num_episodes=100, method='pca')

    # Additional analysis: Correlation between latent dimensions and observation dimensions
    correlations = np.corrcoef(latent_reps.T, observations.T)[:latent_dim, latent_dim:]
    
    print("\nTop correlations between latent dimensions and observation dimensions:")
    for i in range(latent_dim):
        top_corr_idx = np.argmax(np.abs(correlations[i]))
        print(f"Latent dim {i}: Highest correlation {correlations[i, top_corr_idx]:.3f} with obs dim {top_corr_idx}")

    # Visualize these top correlations
    plt.figure(figsize=(15, 10))
    for i in range(5):  # Show top 5 correlations
        latent_dim_i = i
        obs_dim_i = np.argmax(np.abs(correlations[i]))
        plt.subplot(2, 3, i+1)
        plt.scatter(latent_reps[:, latent_dim_i], observations[:, obs_dim_i])
        plt.title(f"Latent dim {latent_dim_i} vs Obs dim {obs_dim_i}")
        plt.xlabel(f"Latent dim {latent_dim_i}")
        plt.ylabel(f"Obs dim {obs_dim_i}")
    plt.tight_layout()
    plt.show()