import os
import time
import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from config import args_ppo
from env.environments import make_env
from models import *
from optimization_utils import *


def train_ppo_agent():
    args_ppo.batch_size = int(args_ppo.num_envs * args_ppo.num_steps)
    args_ppo.minibatch_size = int(args_ppo.batch_size // args_ppo.num_minibatches)
    args_ppo.num_iterations = args_ppo.total_timesteps // args_ppo.batch_size

    random.seed(args_ppo.seed)
    np.random.seed(args_ppo.seed)
    torch.manual_seed(args_ppo.seed)
    torch.backends.cudnn.deterministic = args_ppo.torch_deterministic

    args_ppo.device = torch.device(
        "cuda" if torch.cuda.is_available() and args_ppo.cuda else "cpu"
    )

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args_ppo.env_id,
                i,
                args_ppo.capture_video,
                args_ppo.exp_name,
                args_ppo.gamma,
            )
            for i in range(args_ppo.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    agent = Agent_ppo(envs).to(args_ppo.device)

    if args_ppo.load_model is not None:
        save_dir = os.path.join(os.getcwd(), "params", "ppo")
        data_path = os.path.join(save_dir, args_ppo.load_model)
        if os.path.exists(data_path):
            print(f"Loading model from {data_path}")
            agent.load_state_dict(torch.load(data_path, map_location=args_ppo.device))
        else:
            print(
                f"Model file not found at {data_path}. Starting training from scratch."
            )

    optimizer = optim.Adam(agent.parameters(), lr=args_ppo.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args_ppo.num_steps, args_ppo.num_envs) + envs.single_observation_space.shape
    ).to(args_ppo.device)
    actions = torch.zeros(
        (args_ppo.num_steps, args_ppo.num_envs) + envs.single_action_space.shape
    ).to(args_ppo.device)
    logprobs = torch.zeros((args_ppo.num_steps, args_ppo.num_envs)).to(args_ppo.device)
    rewards = torch.zeros((args_ppo.num_steps, args_ppo.num_envs)).to(args_ppo.device)
    dones = torch.zeros((args_ppo.num_steps, args_ppo.num_envs)).to(args_ppo.device)
    values = torch.zeros((args_ppo.num_steps, args_ppo.num_envs)).to(args_ppo.device)

    # Logging setup
    global_step = 0
    start_time = time.time()
    episodic_returns = []
    episodic_lengths = []
    learning_rates = []
    value_losses = []
    policy_losses = []
    entropies = []
    old_approx_kls = []
    approx_kls = []
    clipfracs = []
    explained_variances = []
    sps_history = []

    next_obs, _ = envs.reset(seed=args_ppo.seed)
    next_obs = torch.Tensor(next_obs).to(args_ppo.device)
    next_done = torch.zeros(args_ppo.num_envs).to(args_ppo.device)

    for iteration in range(1, args_ppo.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args_ppo.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args_ppo.num_iterations
            lrnow = frac * args_ppo.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        learning_rates.append(optimizer.param_groups[0]["lr"])

        for step in range(0, args_ppo.num_steps):
            global_step += args_ppo.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(args_ppo.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(
                args_ppo.device
            ), torch.Tensor(next_done).to(args_ppo.device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        episodic_returns.append(info["episode"]["r"])
                        episodic_lengths.append(info["episode"]["l"])

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(args_ppo.device)
            lastgaelam = 0
            for t in reversed(range(args_ppo.num_steps)):
                if t == args_ppo.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t]
                    + args_ppo.gamma * nextvalues * nextnonterminal
                    - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + args_ppo.gamma
                    * args_ppo.gae_lambda
                    * nextnonterminal
                    * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args_ppo.batch_size)
        clipfracs_batch = []
        for epoch in range(args_ppo.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args_ppo.batch_size, args_ppo.minibatch_size):
                end = start + args_ppo.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs_batch += [
                        ((ratio - 1.0).abs() > args_ppo.clip_coef).float().mean().item()
                    ]

                if args_ppo.target_kl is not None and approx_kl > args_ppo.target_kl:
                    print(
                        f"Early stopping at iteration {iteration} due to reaching target KL."
                    )
                    break

                mb_advantages = b_advantages[mb_inds]
                if args_ppo.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args_ppo.clip_coef, 1 + args_ppo.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Add action regularization term
                action_regularization = (
                    b_actions[mb_inds] ** 2
                ).mean()  # Penalize large actions
                pg_loss += args_ppo.action_reg_coef * action_regularization

                # Value loss
                newvalue = newvalue.view(-1)
                if args_ppo.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args_ppo.clip_coef,
                        args_ppo.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - args_ppo.ent_coef * entropy_loss
                    + v_loss * args_ppo.vf_coef
                    + args_ppo.kl_coef * approx_kl
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args_ppo.max_grad_norm)
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Logging
        value_losses.append(v_loss.item())
        policy_losses.append(pg_loss.item())
        entropies.append(entropy_loss.item())
        old_approx_kls.append(old_approx_kl.item())
        approx_kls.append(approx_kl.item())
        clipfracs.append(np.mean(clipfracs_batch))
        explained_variances.append(explained_var)

        sps = int(global_step / (time.time() - start_time))
        sps_history.append(sps)
        print(f"SPS: {sps}")

    envs.close()

    # Plotting
    plt.figure(figsize=(20, 10))

    plt.subplot(2, 3, 1)
    plt.plot(episodic_returns)
    avg_interval = 50
    # Ensure that metrics["episodic_returns"] is a 1D list or array
    episodic_returns = np.array(episodic_returns).flatten()

    # Now apply np.convolve to calculate the rolling average
    if len(episodic_returns) >= avg_interval:
        avg_returns = np.convolve(
            episodic_returns, np.ones(avg_interval) / avg_interval, mode="valid"
        )
        plt.plot(
            range(avg_interval - 1, len(episodic_returns)),
            avg_returns,
            label=f"{avg_interval}-Episode Average",
            color="orange",
        )

    plt.title("Episodic Returns")
    plt.xlabel("Episode")
    plt.ylabel("Return")

    # plt.subplot(2, 3, 2)
    # plt.plot(episodic_lengths)
    # plt.title('Episodic Lengths')
    # plt.xlabel('Episode')
    # plt.ylabel('Length')

    plt.subplot(2, 3, 2)
    plt.plot(approx_kls)
    plt.title("Approx KLs")
    plt.xlabel("Episode")
    plt.ylabel("Approx KLs")

    plt.subplot(2, 3, 3)
    plt.plot(learning_rates)
    plt.title("Learning Rate")
    plt.xlabel("Iteration")
    plt.ylabel("LR")

    plt.subplot(2, 3, 4)
    plt.plot(value_losses, label="Value Loss")
    plt.plot(policy_losses, label="Policy Loss")
    plt.title("Losses")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(entropies)
    plt.title("Entropy")
    plt.xlabel("Iteration")
    plt.ylabel("Entropy")

    # plt.subplot(2, 3, 6)
    # plt.plot(sps_history)
    # plt.title('Steps Per Second')
    # plt.xlabel('Iteration')
    # plt.ylabel('SPS')

    plt.subplot(2, 3, 6)
    plt.plot(explained_variances)
    plt.title("Explained Variance")
    plt.xlabel("Iteration")
    plt.ylabel("Variance")

    plt.tight_layout()
    plt.savefig("ppo_results.png")
    plt.show()

    save_dir = os.path.join(os.getcwd(), "params","ppo")
    os.makedirs(save_dir, exist_ok=True)

    import re

    existing_files = os.listdir(save_dir)
    run_numbers = [
        int(re.search(r"run_(\d+)", f).group(1))
        for f in existing_files
        if re.search(r"run_(\d+)", f)
    ]
    run_number = max(run_numbers) + 1 if run_numbers else 1

    data_path = os.path.join(save_dir, args_ppo.save_path)
    print("Saved at: ", data_path)
    torch.save(agent.state_dict(), data_path)


if __name__ == "__main__":
    train_ppo_agent()
