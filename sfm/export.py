import os
import torch
import numpy as np
import gymnasium as gym
import random
from ppo import Agent, Args, make_env

# ensured good coordinate

def load_agent(agent_class, path, envs, device):
    agent = agent_class(envs).to(device)
    try:
        agent.load_state_dict(torch.load(path, map_location=device))
        print(f"Successfully loaded model from {path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    agent.eval()
    return agent

def collect_demonstration_data(agent, envs, device, num_episodes=50):
    """Collect demonstration data over multiple episodes"""
    state_history = []
    action_history = []
    next_state_history = []
    total_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = envs.reset(seed=args.seed)
        obs = torch.Tensor(obs).to(device)
        done = False
        episode_reward = 0
        
        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs)
            
            # Store current state and action
            state_history.append(obs.cpu().numpy())
            action_history.append(action.cpu().numpy())
            
            # Step the environment
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminations, truncations)[0]  # Take first element since using vectorized env
            
            # Store next state
            next_state_history.append(next_obs)
            
            # Update for next iteration
            obs = torch.Tensor(next_obs).to(device)
            episode_reward += reward[0]
        
        print(f"Episode {episode + 1} Reward: {episode_reward:.2f}")
        total_rewards.append(episode_reward)
    
    print(f"\nAverage episode reward: {np.mean(total_rewards):.2f}")
    print(f'{np.array(state_history).shape} Experience generated')
    
    return (np.array(state_history), 
            np.array(action_history), 
            np.array(next_state_history))

if __name__ == "__main__":
    # Initialize arguments and set seeds
    args = Args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, args.exp_name, args.gamma) for i in range(args.num_envs)]
    )
    
    # Load model
    model_path = os.path.join(os.getcwd(), 'sfm', 'params', 'ppo/ppo_hc_delay_sensory.pth')
    agent = load_agent(Agent, model_path, envs, device)
    
    if agent is None:
        print("Failed to load agent. Exiting...")
        exit(1)
    
    # Collect demonstration data
    print("\nCollecting demonstration data...")
    states, actions, next_states = collect_demonstration_data(agent, envs, device)
    
    print(f"\nCollected data shapes:")
    print(f"States: {states.shape}")
    print(f"Actions: {actions.shape}")
    print(f"Next States: {next_states.shape}")
    
    # Save the demonstration data
    save_dir = os.path.join(os.getcwd(), 'sfm', 'data')
    os.makedirs(save_dir, exist_ok=True)
    
    data_filename = "imitation_data_ppo_delay.npz"
    npz_path = os.path.join(save_dir, data_filename)
    
    np.savez(npz_path, 
             states=states,
             actions=actions, 
             next_states=next_states)
    
    print(f"\nSaved demonstration data to {npz_path}")
    
    # Close environment
    envs.close()