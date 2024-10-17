import os
import torch
import numpy as np
import gymnasium as gym
from ppo_vector import Agent, Args, make_env  # Import the Agent class, Args, and make_env from the training file

# ensure env did not open noise
target_steps = 10000
collected_steps = 0

# Function to load the saved model
def load_agent(agent_class, path, envs, device):
    agent = agent_class(envs).to(device)
    agent.load_state_dict(torch.load(path))
    agent.eval()  # Set the agent to evaluation mode
    return agent

if __name__ == "__main__":
    # Initialize the arguments (same as in training)
    args = Args()

    # Initialize the environment and agent
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, args.exp_name, args.gamma) for i in range(args.num_envs)]
    )

    # Path to the saved model (adjust the path if necessary)
    model_path = os.path.join(os.getcwd(), 'mvp', 'params', 'ppo_vector_pusher.pth')

    # Load the agent
    agent = load_agent(Agent, model_path, envs, device)

    # Data logging for states, actions, and next_states
    state_history = []
    action_history = []
    next_state_history = []

    # Run a test episode and record states, actions, next_states
    obs, _ = envs.reset(seed=args.seed)
    obs = torch.Tensor(obs).to(device)
    done = False

    while collected_steps < target_steps:
        done = False
        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs)

            # Log the current state and action
            state_history.append(obs.cpu().numpy())
            action_history.append(action.cpu().numpy())

            # Step the environment
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminations, truncations)

            # Log the next state
            next_state_history.append(next_obs)

            obs = torch.Tensor(next_obs).to(device)
            collected_steps += 1

            if collected_steps >= target_steps:
                break

    # Convert to numpy arrays
    state_history = np.array(state_history)
    action_history = np.array(action_history)
    next_state_history = np.array(next_state_history)

    # Save the logged data into a single .npz file for later analysis
    save_dir = os.path.join(os.getcwd(), 'mvp', 'data')
    os.makedirs(save_dir, exist_ok=True)

    data_filename = "imitation_data_pusher.npz"
    npz_path = os.path.join(save_dir, data_filename)
    np.savez(npz_path, states=state_history, actions=action_history, next_states=next_state_history)

    print(f"Logged and saved state, action, and next_state data in {npz_path}")

    # Close the environment properly
    try:
        envs.close()
    except Exception as e:
        print(f"Error during environment closure: {e}")