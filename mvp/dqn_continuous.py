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

# from mvp.dqn_networks import ReplayMemory, DQN, Transition
from mvp.env_pendulum import PendulumEnv

is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

BATCH_SIZE = 200
GAMMA = 0.98
EPS_START = 1
EPS_END = 0.001
EPS_DECAY = 0.98
TAU = 0.01
LR = 0.01
BUFFER = 100000

register(
    id="Pendulum-v0", entry_point="mvp.env_pendulum:PendulumEnv", max_episode_steps=300
)
env = gym.make("Pendulum-v0")

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        # if n_actions is different, matrix mismatch
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

n_actions = env.action_space.shape[0]
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
    """Continuous action float output"""
    # print(state)
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            # return policy_net(state).max(1).indices.view(1, 1)
            # return policy_net(state).max(1).values.unsqueeze(1)
            return policy_net(state)  # policy net give torque directly
    else:
        # return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        return torch.tensor(
            [[env.action_space.sample()]], device=device, dtype=torch.float
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
    next_state_batch = torch.cat([s for s in batch.next_state if s is not None])

    # Separate states and next states (assuming single state/next state per transition)
    state_batch = torch.cat(batch.state)

    action_batch = torch.tensor(
        [t.action for t in transitions], device=device, dtype=torch.float
    )
    # squeezed_action = tuple([b.squeeze().unsqueeze(0).unsqueeze(0) for b in batch.action])
    # action_batch = torch.cat(squeezed_action)

    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - directly get Q values from the policy network
    state_action_values = policy_net(state_batch)  # .gather(1,action_batch)

    # Compute V(s_{t+1}) for non-final states using target network
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(next_state_batch).max(1).values
        # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    # Expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def load_model_and_evaluate(path, device="cpu"):
    """Evaluate the model"""
    env = PendulumEnv(render_mode="human")
    n_actions = env.action_space.shape[0]
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

        with torch.no_grad():
            action = select_action(state)

        action = action.squeeze(0)
        observation, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            print(f"Episode finished after {t+1} timesteps")
            break

        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(
            0
        )

    env.close()


def main():
    """Main training loop"""
    num_episodes = 500 if torch.cuda.is_available() else 300

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            action = select_action(state)
            # print(action.squeeze().unsqueeze(0))
            observation, reward, terminated, truncated, _ = env.step(
                action.squeeze().unsqueeze(0)
            )  # pass in as a 1d

            reward = torch.tensor([reward], device=device)

            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

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

    torch.save(policy_net.state_dict(), "params/pendulum_dqn_continuous.pth")
    print("Complete")
    plot_reward(show_result=True)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
