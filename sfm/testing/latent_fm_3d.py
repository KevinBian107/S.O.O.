import os
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sfm.sfmppo import Args, UPN, make_env
from mpl_toolkits.mplot3d import Axes3D

def extract_latent_representations(upn_model, envs, device, num_episodes=10):
    '''latent representation is z, action performed is random in here'''
    latent_reps = []
    episode_returns = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, _ = envs.reset()
        done = np.zeros(envs.num_envs, dtype=bool)
        episode_return = 0
        episode_length = 0
        
        while not done.all():
            obs_tensor = torch.FloatTensor(obs).to(device)
            with torch.no_grad():
                latent = upn_model.encoder(obs_tensor)
                latent_reps.append(latent.cpu().numpy())
            
            action = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            obs, reward, terminated, truncated, _ = envs.step(action)
            episode_return += reward[0]  # Assuming single environment for simplicity
            done = np.logical_or(terminated, truncated)
            episode_length += 1
        
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}/{num_episodes}, Return: {episode_return}, Length: {episode_length}")
    
    return np.vstack(latent_reps), episode_returns, episode_lengths

def reduce_dimensionality(latent_reps, method='pca', n_components=3):
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, 
                       perplexity=30, 
                       n_iter=500, 
                       learning_rate=200, 
                       random_state=42)
    else:
        raise ValueError("Method must be either 'pca' or 'tsne'")
    
    return reducer.fit_transform(latent_reps)

def visualize_latent_space_3d(reduced_reps, episode_returns, episode_lengths, method):
    fig = plt.figure(figsize=(20, 10))

    # Plot latent space colored by episode return
    ax1 = fig.add_subplot(121, projection='3d')
    return_colors = np.repeat(episode_returns, episode_lengths)
    scatter = ax1.scatter(reduced_reps[:, 0], reduced_reps[:, 1], reduced_reps[:, 2], 
                          c=return_colors, cmap='viridis', alpha=0.5)
    fig.colorbar(scatter, ax=ax1, label='Episode Return')
    ax1.set_title(f'3D Latent Space Visualization using {method.upper()} (colored by return)')
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('Component 2')
    ax1.set_zlabel('Component 3')

    # Plot latent space colored by step within episode
    ax2 = fig.add_subplot(122, projection='3d')
    step_colors = np.concatenate([np.arange(length) for length in episode_lengths])
    scatter = ax2.scatter(reduced_reps[:, 0], reduced_reps[:, 1], reduced_reps[:, 2], 
                          c=step_colors, cmap='coolwarm', alpha=0.5)
    fig.colorbar(scatter, ax=ax2, label='Step within Episode')
    ax2.set_title(f'3D Latent Space Visualization using {method.upper()} (colored by step)')
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    ax2.set_zlabel('Component 3')

    plt.tight_layout()
    plt.show()

def analyze_latent_space(upn_model, envs, device, num_episodes=10, method='pca'):
    latent_reps, episode_returns, episode_lengths = extract_latent_representations(upn_model, envs, device, num_episodes)
    reduced_reps = reduce_dimensionality(latent_reps, method, n_components=3)
    visualize_latent_space_3d(reduced_reps, episode_returns, episode_lengths, method)
    return latent_reps, reduced_reps, episode_returns, episode_lengths

if __name__ == "__main__":
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, args.exp_name, args.gamma) for i in range(args.num_envs)]
    )

    # Create and load only the UPN model
    state_dim = np.array(envs.single_observation_space.shape).prod()
    action_dim = np.prod(envs.single_action_space.shape)
    latent_dim = 32  # Same as FM-PPO latent dimension
    upn_model = UPN(state_dim, action_dim, latent_dim).to(device)

    fm_path = os.path.join(os.getcwd(), "mvp", "params", "fm_vector.pth")
    fm_state_dict = torch.load(fm_path, map_location=device)
    upn_model.load_state_dict(fm_state_dict)

    # Analyze latent space
    latent_reps, reduced_reps, episode_returns, episode_lengths = analyze_latent_space(upn_model, envs, device, num_episodes=100, method='tsne')

    # Additional analysis: Correlation between latent dimensions and episode returns
    episode_latents = np.array([np.mean(latent_reps[sum(episode_lengths[:i]):sum(episode_lengths[:i+1])], axis=0) for i in range(len(episode_returns))])
    correlations = np.corrcoef(episode_latents.T, episode_returns)[:-1, -1]
    top_corr_dims = np.argsort(np.abs(correlations))[-5:][::-1]
    
    # print("\nTop 5 latent dimensions correlated with episode returns:")
    # for dim in top_corr_dims:
    #     print(f"Dimension {dim}: Correlation {correlations[dim]:.3f}")

    # # Visualize these top dimensions
    # plt.figure(figsize=(15, 10))
    # for i, dim in enumerate(top_corr_dims):
    #     plt.subplot(2, 3, i+1)
    #     plt.scatter(episode_latents[:, dim], episode_returns)
    #     plt.title(f"Dimension {dim}")
    #     plt.xlabel("Mean Latent Value")
    #     plt.ylabel("Episode Return")
    # plt.tight_layout()
    # plt.show()