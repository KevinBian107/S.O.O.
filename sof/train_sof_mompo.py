import os
import time
import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from config import args_sof
from environments import make_env
from models import *
from optimization_utils import *


def train_sofppo_agent():
    args_sof.batch_size = args_sof.num_steps * args_sof.num_envs
    args_sof.minibatch_size = args_sof.batch_size // args_sof.num_minibatches
    args_sof.iterations = args_sof.total_timesteps // args_sof.batch_size

    random.seed(args_sof.seed)
    np.random.seed(args_sof.seed)
    torch.manual_seed(args_sof.seed)
    torch.backends.cudnn.deterministic = args_sof.torch_deterministic

    args_sof.device = torch.device(
        "cuda" if torch.cuda.is_available() and args_sof.cuda else "cpu"
    )

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args_sof.env_id,
                i,
                args_sof.capture_video,
                args_sof.exp_name,
                args_sof.gamma,
            )
            for i in range(args_sof.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    agent = Agent_sof(envs).to(args_sof.device)

    # freeze_base_controller(agent)

    if args_sof.load_sfmppo is not None:
        save_dir = os.path.join(os.getcwd(), "sof", "params", "sofppo")
        data_path = os.path.join(save_dir, args_sof.load_sfmppo)
        if os.path.exists(data_path):
            print(f"Loading sfmppo model from {data_path}")
            agent.load_ppo(data_path)  # Use the new method to load only PPO parameters
        else:
            print(
                f"Model file not found at {data_path}. Starting training from scratch."
            )

    if args_sof.load_upn is not None:
        # if args_sof.load_sfmppo is not None:
        #     print('Loading Full model, cannot load sfm core')
        # else:
        # Define the path to save and load UPN weights
        print("loaded params for supervised forward model")
        model_dir = os.path.join(os.getcwd(), "sof", "params", "supp")
        os.makedirs(model_dir, exist_ok=True)
        load_path = os.path.join(model_dir, args_sof.load_upn)
        # Attempt to load UPN weights
        agent.load_upn(load_path)

    # Optimizer for PPO (actor and critic)
    ppo_optimizer = optim.Adam(
        [
            {"params": agent.actor_mean.parameters()},
            {"params": agent.actor_logstd},
            {"params": agent.critic.parameters()},
        ],
        lr=args_sof.ppo_learning_rate,
        eps=1e-5,
    )

    # Optimizer for UPN
    upn_optimizer = optim.Adam(
        agent.upn.parameters(), lr=args_sof.upn_learning_rate, eps=1e-5
    )

    eta_optimizer = optim.Adam([agent.eta_k], lr=args_sof.eta_learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args_sof.num_steps, args_sof.num_envs) + envs.single_observation_space.shape
    ).to(args_sof.device)
    actions = torch.zeros(
        (args_sof.num_steps, args_sof.num_envs) + envs.single_action_space.shape
    ).to(args_sof.device)
    logprobs = torch.zeros((args_sof.num_steps, args_sof.num_envs)).to(args_sof.device)
    rewards = torch.zeros((args_sof.num_steps, args_sof.num_envs)).to(args_sof.device)
    dones = torch.zeros((args_sof.num_steps, args_sof.num_envs)).to(args_sof.device)
    values = torch.zeros((args_sof.num_steps, args_sof.num_envs)).to(args_sof.device)
    # this is only for upn
    next_obs_all = torch.zeros(
        (args_sof.num_steps, args_sof.num_envs) + envs.single_observation_space.shape
    ).to(args_sof.device)

    # Logging setup
    global_step = 0
    start_time = time.time()
    metrics = {
        "episodic_returns": [],
        "episodic_lengths": [],
        "learning_rates": [],
        "value_losses": [],
        "policy_losses": [],
        "entropies": [],
        "approx_kls": [],
        "clipfracs": [],
        "explained_variances": [],
        "upn_losses": [],
        "recon_losses": [],
        "forward_losses": [],
        "inverse_losses": [],
        "consist_losses": [],
        "kl_constrained_penalty": [],
        "eta_k_loss": [],
    }

    next_obs, _ = envs.reset(seed=args_sof.seed)
    next_obs = torch.Tensor(next_obs).to(args_sof.device)
    next_done = torch.zeros(args_sof.num_envs).to(args_sof.device)

    for iteration in range(1, args_sof.iterations + 1):
        if args_sof.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args_sof.iterations
            lrnow = frac * args_sof.ppo_learning_rate
            ppo_optimizer.param_groups[0]["lr"] = lrnow

        metrics["learning_rates"].append(ppo_optimizer.param_groups[0]["lr"])

        for step in range(0, args_sof.num_steps):
            global_step += args_sof.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(args_sof.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(
                args_sof.device
            ), torch.Tensor(next_done).to(args_sof.device)
            next_obs_all[step] = next_obs

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        metrics["episodic_returns"].append(info["episode"]["r"])
                        metrics["episodic_lengths"].append(info["episode"]["l"])

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(args_sof.device)
            lastgaelam = 0
            for t in reversed(range(args_sof.num_steps)):
                if t == args_sof.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t]
                    + args_sof.gamma * nextvalues * nextnonterminal
                    - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + args_sof.gamma
                    * args_sof.gae_lambda
                    * nextnonterminal
                    * lastgaelam
                )
            returns = advantages + values

        if args_sof.mix_coord:
            # mixing screw things up, isolate the problem bit by bit
            obs_imitate, actions_imitate, next_obs_imitate = mixed_batch(
                obs, actions, next_obs_all
            )
        else:
            obs_imitate, actions_imitate, next_obs_imitate = obs, actions, next_obs_all

        # Mixed batch with imitation data
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # imitate mix
        b_obs_imitate = obs_imitate.reshape((-1,) + envs.single_observation_space.shape)
        b_actions_imitate = actions_imitate.reshape(
            (-1,) + envs.single_action_space.shape
        )
        b_next_obs_imitate = next_obs_imitate.reshape(
            (-1,) + envs.single_observation_space.shape
        )  # previous error of passing the same obs help may be due to having 2 obs in action selection

        b_inds = np.arange(args_sof.batch_size)
        clipfracs_batch = []
        for epoch in range(args_sof.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args_sof.batch_size, args_sof.minibatch_size):
                end = start + args_sof.minibatch_size
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
                        ((ratio - 1.0).abs() > args_sof.clip_coef).float().mean().item()
                    ]

                # if args_sof.target_kl is not None and approx_kl > args_sof.target_kl:
                #     print(f"Early stopping at iteration {iteration} due to reaching target KL.")
                #     break

                mb_advantages = b_advantages[mb_inds]
                if args_sof.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args_sof.clip_coef, 1 + args_sof.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args_sof.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args_sof.clip_coef,
                        args_sof.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                eta_loss = compute_eta_k_loss(agent, b_advantages, args_sof.epsilon_k)

                # Lagrangian Objective (Adjusted with KL Hidden Distribution Constraint)
                hidden_dist = compute_hidden_action_distribution(
                    agent,
                    b_obs_imitate[mb_inds],
                    b_advantages[mb_inds],
                    args_sof.epsilon_k,
                    agent.eta_k,
                )
                kl_constraint_penalty = compute_lagrangian_kl_constraint(
                    agent,
                    b_obs_imitate[mb_inds],
                    agent.eta_k,
                    args_sof.epsilon_k,
                    hidden_dist,
                )
                recon_loss, forward_loss, inverse_loss, consistency_loss = (
                    compute_upn_loss(
                        agent.upn,
                        b_obs_imitate[mb_inds],
                        b_actions_imitate[mb_inds],
                        b_next_obs_imitate[mb_inds],
                    )
                )
                ppo_loss = (
                    pg_loss
                    - args_sof.ent_coef * entropy_loss
                    + v_loss * args_sof.vf_coef
                    + approx_kl * args_sof.kl_coef
                    + kl_constraint_penalty * args_sof.constrain_weights
                )
                # Previously not on in sfmppo
                upn_loss = args_sof.upn_coef * (
                    recon_loss + forward_loss + inverse_loss + consistency_loss
                )

                # PPO backward pass and optimization
                ppo_optimizer.zero_grad()
                ppo_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(agent.actor_mean.parameters())
                    + [agent.actor_logstd]
                    + list(agent.critic.parameters()),
                    args_sof.max_grad_norm,
                )
                ppo_optimizer.step()

                # UPN backward pass and optimization
                upn_optimizer.zero_grad()
                upn_loss.backward()
                nn.utils.clip_grad_norm_(agent.upn.parameters(), args_sof.max_grad_norm)
                upn_optimizer.step()

                # Only backpropagate the KL penalty through eta_k
                eta_optimizer.zero_grad()
                eta_loss.backward()
                eta_optimizer.step()

                # Clip eta_k to be positive
                with torch.no_grad():
                    agent.eta_k.clamp_(min=1e-5)

                for name, param in agent.named_parameters():
                    if param.grad is not None and (
                        torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                    ):
                        print(f"NaN or Inf detected in gradients of {name}")

                # grad_norms = []
                # for name, param in agent.named_parameters():
                #     if param.grad is not None:
                #         grad_norms.append(param.grad.norm().item())
                # print("Max grad norm:", max(grad_norms))

                # nn.utils.clip_grad_norm_(agent.parameters(), args_sof.max_grad_norm)
                # # optimizer.step()

        # Logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        metrics["value_losses"].append(v_loss.item())
        metrics["policy_losses"].append(pg_loss.item())
        metrics["upn_losses"].append(upn_loss.item())
        metrics["forward_losses"].append(forward_loss.item())
        metrics["inverse_losses"].append(inverse_loss.item())
        metrics["recon_losses"].append(recon_loss.item())
        metrics["consist_losses"].append(consistency_loss.item())
        metrics["entropies"].append(entropy_loss.item())
        metrics["approx_kls"].append(approx_kl.item())
        metrics["kl_constrained_penalty"].append(kl_constraint_penalty.item())
        metrics["eta_k_loss"].append(eta_loss.item())
        metrics["clipfracs"].append(np.mean(clipfracs_batch))
        metrics["explained_variances"].append(explained_var)

        sps = int(global_step / (time.time() - start_time))
        print(f"SPS: {sps}")

    envs.close()

    # Plotting results
    plt.figure(figsize=(20, 10))

    plt.subplot(2, 3, 1)
    plt.plot(metrics["episodic_returns"])

    avg_interval = args_sof.graph_avg_interval
    # Ensure that metrics["episodic_returns"] is a 1D list or array
    episodic_returns = np.array(metrics["episodic_returns"]).flatten()

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
    # plt.plot(metrics["episodic_lengths"])
    # plt.title('Episodic Lengths')
    # plt.xlabel('Episode')
    # plt.ylabel('Length')

    plt.subplot(2, 3, 2)
    plt.plot(metrics["approx_kls"])
    plt.title("Approx KLs")
    plt.xlabel("Episode")
    plt.ylabel("Approx KLs")

    # plt.subplot(2, 3, 3)
    # plt.plot(metrics["learning_rates"])
    # plt.title('Learning Rate')
    # plt.xlabel('Iteration')
    # plt.ylabel('LR')

    plt.subplot(2, 3, 3)
    plt.plot(metrics["kl_constrained_penalty"])
    plt.title("KL Constraint Penalty")
    plt.xlabel("Iteration")
    plt.ylabel("KL-CP")

    plt.subplot(2, 3, 4)
    plt.plot(metrics["value_losses"], label="Value Loss")
    plt.plot(metrics["policy_losses"], label="Policy Loss")
    plt.plot(metrics["upn_losses"], label="UPN Loss")
    plt.plot(metrics["forward_losses"], label="Forward Loss")
    plt.plot(metrics["inverse_losses"], label="Inverse Loss")
    plt.plot(metrics["recon_losses"], label="Reconstruction Loss")
    plt.plot(metrics["consist_losses"], label="Consistency Loss")
    plt.plot(metrics["eta_k_loss"], label="Eta K Loss")
    plt.title("Losses")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(metrics["entropies"])
    plt.title("Entropy")
    plt.xlabel("Iteration")
    plt.ylabel("Entropy")

    plt.subplot(2, 3, 6)
    plt.plot(metrics["explained_variances"])
    plt.title("Explained Variance")
    plt.xlabel("Iteration")
    plt.ylabel("Variance")

    plt.tight_layout()
    plt.savefig("sofppo_vae_constrained.png")
    plt.show()

    # Save the model
    save_dir1 = os.path.join(os.getcwd(), "sof", "params", "sofppo")
    save_dir2 = os.path.join(os.getcwd(), "sof", "params", "sof")
    os.makedirs(save_dir, exist_ok=True)

    data1_path = os.path.join(save_dir1, args_sof.save_sfmppo)
    data2_path = os.path.join(save_dir2, args_sof.save_sfm)

    print("Saved at: ", data1_path)
    torch.save(agent.state_dict(), data1_path)

    print("Saved at: ", data2_path)
    torch.save(agent.upn.state_dict(), data2_path)


if __name__ == "__main__":
    train_sofppo_agent()
