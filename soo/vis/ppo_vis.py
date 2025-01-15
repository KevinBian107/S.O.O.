import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from config import args_test
from sof.env.environments import make_env
from models import Agent_ppo as PPOAgent


class FullPPOActivationVisualizer:
    def __init__(self, agent, envs, device, method="pca", fig_size=(20, 10)):
        self.agent = agent
        self.envs = envs
        self.device = device
        self.method = method
        self.fig_size = fig_size

        # Initialize visualization
        plt.ion()
        self.fig = plt.figure(figsize=self.fig_size)

        # Create subplots: env view, actor activation space, critic activation space, action space
        self.ax_env = self.fig.add_subplot(231)
        self.ax_actor = self.fig.add_subplot(232)
        self.ax_critic = self.fig.add_subplot(233)
        self.ax_action = self.fig.add_subplot(234)

        # Initialize PCAs for all spaces
        self.actor_pca = PCA(n_components=2)
        self.critic_pca = PCA(n_components=2)
        self.action_pca = PCA(n_components=2)

        self.trajectories = {"actor": [], "critic": [], "action": []}

        self.trajectory_lines = {"actor": None, "critic": None, "action": None}

        # Get initial frame
        obs, _ = self.envs.reset()
        frame = self.envs.call("render")[0]
        self.frame_shape = frame.shape

    def get_network_activations(self, obs):
        """Extract intermediate activations from actor and critic networks"""
        with torch.no_grad():
            # Actor activation
            x = obs
            for i, layer in enumerate(self.agent.actor_mean):
                x = layer(x)
                if i == len(self.agent.actor_mean) - 2:  # Second-to-last layer
                    actor_hidden = x

            # Critic activation
            critic_hidden = self.agent.critic[0](obs)

        return {"actor": actor_hidden, "critic": critic_hidden}

    def collect_initial_representations(self, num_episodes=5):
        """Collect initial actor, critic, and action representations"""
        collections = {
            "actor": [],
            "critic": [],
            "actions": [],
            "returns": [],
            "steps": [],
        }

        print("Collecting initial representations...")
        for episode in range(num_episodes):
            next_obs, _ = self.envs.reset()
            next_obs = torch.Tensor(next_obs).to(self.device)
            next_done = torch.zeros(self.envs.num_envs).to(self.device)
            episode_return = torch.zeros(self.envs.num_envs).to(self.device)
            step_count = 0

            while not next_done.all():
                with torch.no_grad():
                    # Get actor and critic activations
                    activations = self.get_network_activations(next_obs)
                    for key, value in activations.items():
                        collections[key].append(value.cpu().numpy())

                    # Get action
                    action, _, _, _ = self.agent.get_action_and_value(next_obs)
                    collections["actions"].append(action.cpu().numpy())

                    # Step environment
                    next_obs, reward, terminations, truncations, _ = self.envs.step(
                        action.cpu().numpy()
                    )
                    next_obs = torch.Tensor(next_obs).to(self.device)
                    next_done = torch.logical_or(
                        torch.Tensor(terminations), torch.Tensor(truncations)
                    ).to(self.device)
                    episode_return += torch.Tensor(reward).to(self.device) * (
                        ~next_done
                    )
                    step_count += 1

            print(
                f"Episode {episode + 1}/{num_episodes} completed with return: {episode_return.item()}"
            )
            collections["returns"].append(episode_return.item())
            collections["steps"].append(step_count)

        return collections

    def setup_visualization(self):
        """Initialize visualization with collected data"""
        collections = self.collect_initial_representations()

        # Fit PCAs
        reduced_data = {
            "actor": self.actor_pca.fit_transform(np.vstack(collections["actor"])),
            "critic": self.critic_pca.fit_transform(np.vstack(collections["critic"])),
            "action": self.action_pca.fit_transform(np.vstack(collections["actions"])),
        }

        # Clear all axes
        for ax in [self.ax_env, self.ax_actor, self.ax_critic, self.ax_action]:
            ax.clear()

        # Create scatter plots for each space
        axes_map = {
            "actor": self.ax_actor,
            "critic": self.ax_critic,
            "action": self.ax_action,
        }

        self.current_points = {}

        for space, ax in axes_map.items():
            # Background scatter
            scatter = ax.scatter(
                reduced_data[space][:, 0],
                reduced_data[space][:, 1],
                c=np.repeat(collections["returns"], collections["steps"]),
                cmap="viridis",
                alpha=0.5,
            )
            plt.colorbar(scatter, ax=ax, label="Episode Return")

            # Current point
            self.current_points[space] = ax.scatter(
                [], [], c="red", s=200, label="Current state"
            )

            # Trajectory line
            (self.trajectory_lines[space],) = ax.plot(
                [], [], "r-", alpha=0.5, linewidth=3
            )

            # Add explained variance ratio to titles
            var = getattr(self, f"{space}_pca").explained_variance_ratio_
            ax.set_title(f"{space.capitalize()} Space\nVar: {var[0]:.2f}, {var[1]:.2f}")
            ax.legend()

        # Initialize environment display
        self.env_image = self.ax_env.imshow(np.zeros(self.frame_shape))
        self.ax_env.set_title("Environment")
        self.ax_env.axis("off")

        plt.tight_layout()
        self.fig.canvas.draw()
        print("Visualization setup completed")

    def update_visualization(self, obs, episode_return):
        """Update visualizations with current state"""
        try:
            # Get current representations
            with torch.no_grad():
                activations = self.get_network_activations(
                    torch.Tensor(obs).to(self.device)
                )
                action, _, _, _ = self.agent.get_action_and_value(
                    torch.Tensor(obs).to(self.device)
                )

                current_points = {
                    "actor": self.actor_pca.transform(
                        activations["actor"].cpu().numpy()
                    ),
                    "critic": self.critic_pca.transform(
                        activations["critic"].cpu().numpy()
                    ),
                    "action": self.action_pca.transform(action.cpu().numpy()),
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
            frame = self.envs.call("render")[0]
            if frame is not None:
                self.env_image.set_array(frame)

            # Update title with current return
            self.ax_env.set_title(f"Environment (Return: {episode_return:.2f})")

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
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.envs.num_envs).to(self.device)
        episode_return = torch.zeros(self.envs.num_envs).to(self.device)
        step_count = 0

        print("Starting new episode...")
        self.trajectories = {key: [] for key in self.trajectories.keys()}

        while not next_done.all():
            self.update_visualization(next_obs.cpu().numpy(), episode_return.item())

            with torch.no_grad():
                action, _, _, _ = self.agent.get_action_and_value(next_obs)

            next_obs, reward, terminations, truncations, _ = self.envs.step(
                action.cpu().numpy()
            )
            next_obs = torch.Tensor(next_obs).to(self.device)
            next_done = torch.logical_or(
                torch.Tensor(terminations), torch.Tensor(truncations)
            ).to(self.device)
            episode_return += torch.Tensor(reward).to(self.device) * (~next_done)
            step_count += 1

            if step_count % 100 == 0:
                print(
                    f"Steps: {step_count}, Current Return: {episode_return.item():.2f}"
                )

        final_return = episode_return.item()
        print(f"Episode completed - Steps: {step_count}, Return: {final_return:.2f}")
        return final_return


def main():
    print("Initializing environment and agent...")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args_test.cuda else "cpu"
    )
    print(f"Using device: {device}")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args_test.env_id, i, True, args_test.exp_name, args_test.gamma)
            for i in range(1)
        ]
    )
    print(f"Environment created: {args_test.env_id}")

    agent = PPOAgent(envs).to(device)
    model_path = os.path.join(os.getcwd(), "sof", "params", "ppo", args_test.ppo_path)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully")

    visualizer = FullPPOActivationVisualizer(agent, envs, device)
    visualizer.setup_visualization()

    num_episodes = 5
    returns = []
    for episode in range(num_episodes):
        print(f"\nStarting Episode {episode + 1}/{num_episodes}")
        episode_return = visualizer.run_episode()
        returns.append(episode_return)
        print(
            f"Episode {episode + 1}/{num_episodes} completed, Return: {episode_return:.2f}"
        )

    print("\nAll episodes completed")
    print(f"Average Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
