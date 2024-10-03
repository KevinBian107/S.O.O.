import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
from gym.envs.registration import register

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from dqn_networks import ReplayMemory, DQN, Transition
from env_pendulum import PendulumEnv

is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

BATCH_SIZE = 200
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.001
EPS_DECAY = 0.98
TAU = 0.01
LR = 0.00001
BUFFER = 100000

register(
    id="Pendulum-v0", entry_point="env_pendulum:PendulumEnv", max_episode_steps=300
)
env = gym.make("Pendulum-v0")


ACTION_MAP = np.linspace(-2, 2, 5)  # 5 actions ranging from -2 to 2
n_actions = len(ACTION_MAP)
# n_actions = env.action_space.shape[0]
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(BUFFER)

steps_done = 0
episode_rewards = []


def select_action(state):
    """Continuous action float output, maybe discretize action?"""
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)  # single index
            # return policy_net(state).squeeze().unsqueeze(0) # policy net give torque directly
    else:
        # return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.float).squeeze().unsqueeze(0)
        return torch.tensor(
            [[random.randrange(n_actions)]], device=device, dtype=torch.long
        )


def plot_reward(show_result=False):
    """Plot the reward achieved"""

    plt.figure(1)
    reward_episode = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("rwards")
    plt.plot(reward_episode.numpy())
    # Take 100 episode averages and plot them too
    if len(reward_episode) >= 100:
        means = reward_episode.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    """optimization steps"""
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch & convert batch-array of Transitions to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )

    # Separate states and next states (assuming single state/next state per transition)
    next_state_batch = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    reward_batch = torch.cat(batch.reward)

    action_batch = torch.tensor(
        [t.action for t in transitions], device=device, dtype=torch.float
    )
    action_batch = action_batch.type(torch.int64).unsqueeze(
        1
    )  # need to match 2d tensor

    # Compute Q(s_t, a) - directly get Q values from the policy network
    q_values = policy_net(state_batch)
    # print(q_values.shape, action_batch.shape)

    if (action_batch < 0).any() or (action_batch >= 5).any():
        raise ValueError("Action batch contains out-of-bounds values")

    state_action_values = q_values.gather(1, action_batch)

    # Compute V(s_{t+1}) for non-final states using target network
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(next_state_batch).max(1).values

    # Expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1)
    optimizer.step()


def load_model_and_evaluate(path, device="cpu"):
    """Evaluate the model"""
    env = PendulumEnv(render_mode="human")
    ACTION_MAP = np.linspace(-2, 2, 5)  # 5 actions ranging from -2 to 2
    n_actions = len(ACTION_MAP)
    state, info = env.reset()
    n_observations = len(state)

    model = DQN(n_observations, n_actions).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    num_eval_episodes = 10
    for i_episode in range(num_eval_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            env.render()

            action_idx = model(state).max(1)[1]
            actual_action = ACTION_MAP[action_idx.item()]

            observation, reward, terminated, truncated, _ = env.step([actual_action])
            if terminated or truncated:
                print(f"Episode finished after {t+1} timesteps")
                break

            state = torch.tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)

    env.close()


def main(checkpoint_path=None):
    """Main training loop, only load in policy_net weight"""

    num_episodes = 1500 if torch.cuda.is_available() else 1200
    num_episodes = 300 if checkpoint_path else 1200

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        policy_net.load_state_dict(checkpoint)
        # target_net.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            action_idx = select_action(state)

            actual_action = ACTION_MAP[action_idx.item()]

            observation, reward, terminated, truncated, _ = env.step(
                [actual_action]
            )  # pass in as a 1d

            reward = torch.tensor([reward], device=device)

            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # Store the transition in memory, should push in action as the index, not actual values
            memory.push(state, action_idx.item(), next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights instead of directly learning the weight, intermediate learning
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_rewards.append(reward)
                plot_reward()
                break

    if checkpoint_path:
        torch.save(policy_net.state_dict(), "params/pendulum_dqn_discrete_retrain.pth")
    else:
        torch.save(policy_net.state_dict(), "params/pendulum_dqn_discrete.pth")

    print("Complete")
    plot_reward(show_result=True)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
