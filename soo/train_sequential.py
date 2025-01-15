import os
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt

from train_soo import train_sofppo_agent
from env.env_wrappers import (
    TargetVelocityWrapper,
    JumpRewardWrapper,
    FlipRewardWrapper
)
from env.environments import make_env_with_wrapper
from config import args_sof
from models import Agent_sof

TASKS = [
    {
        "env_id": "HalfCheetah-v4",
        "name": "Normal Env",
        "wrappers": [lambda env: env],
    },
    {
        "env_id": "HalfCheetah-v4",
        "name": "Target Velocity = 2.0",
        "wrappers": [lambda env: TargetVelocityWrapper(env, target_velocity=2.0)],
    },
    {
        "env_id": "HalfCheetah-v4",
        "name": "Target Jump = 1.5",
        "wrappers": [lambda env: JumpRewardWrapper(env, jump_target_height=1.5)],
    },
    {
        "env_id": "HalfCheetah-v4",
        "name": "Flipping Reward",
        "wrappers": [lambda env: FlipRewardWrapper(env)],
    },
]

def evaluate_agent(agent, env_id, wrappers, num_episodes=5, device="cpu"):
    """
    Runs `num_episodes` of evaluation on a given environment and returns the average episodic return.
    """
    returns = []
    for _ in range(num_episodes):
        # Build a *single* (non-vectorized) environment instance for eval
        env = make_env_with_wrapper(
            env_id=env_id,
            idx=0,
            capture_video=False,
            run_name="eval_temp",
            gamma=args_sof.gamma,
            wrappers=wrappers
        )()
        obs, _ = env.reset()
        done = False
        episode_return = 0.0

        # Convert obs to torch if your agent expects that
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_t.unsqueeze(0))
            action_np = action.cpu().numpy()[0]
            
            obs_next, reward, terminated, truncated, info = env.step(action_np)
            obs_t = torch.as_tensor(obs_next, dtype=torch.float32, device=device)
            done = terminated or truncated
            episode_return += reward

        env.close()
        returns.append(episode_return)
    return np.mean(returns)

def plot_retention_curves(retention_log, current_task_index):
    """
    Plot how performance on each environment changes after finishing each task.
    `retention_log` is a dict like:
      {
        "env_0": [perf_after_task0, perf_after_task1, ...],
        "env_1": [...],
        ...
      }
    """
    plt.figure(figsize=(7, 5))
    for env_key, perf_list in retention_log.items():
        plt.plot(perf_list, marker='o', label=env_key)

    plt.title(f"Retention after Task {current_task_index}")
    plt.xlabel("Task Index (training finished)")
    plt.ylabel("Average Episodic Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    img_dir = os.path.join(os.getcwd(), "retention_imgs")
    os.makedirs(img_dir, exist_ok=True)
    
    plt.savefig(os.path.join(img_dir, f"retention_plot_task_{current_task_index}.png"))
    plt.close()
    print(f"[INFO] Saved retention plot: retention_plot_task_{current_task_index}.png")


def train_multiple_tasks_sequentially():
    multi_task_save_dir = os.path.join(os.getcwd(), "params", "multi_task")
    os.makedirs(multi_task_save_dir, exist_ok=True)

    for i, task_info in enumerate(TASKS):
        env_id = task_info["env_id"]
        task_name = task_info["name"]
        wrappers = task_info["wrappers"]
        print(wrappers)

        # Build a vectorized environment for the current task.
        cur_env = gym.vector.SyncVectorEnv(
            [
                make_env_with_wrapper(
                    env_id=env_id,
                    idx=env_idx,
                    capture_video=args_sof.capture_video,
                    run_name=args_sof.exp_name,
                    gamma=args_sof.gamma,
                    wrappers=wrappers,
                )
                for env_idx in range(args_sof.num_envs)
            ]
        )

        print(f"\n=== Training on Task #{i} ===")
        print(f"Task name: {task_name} (env_id={env_id})")

        if i > 0:
            prev_core_path = os.path.join(
                multi_task_save_dir, f"multi_task/soo_core_task_{i-1}.pth"
            )
            args_sof.load_sfmppo = None
            args_sof.load_sfmppo = os.path.basename(prev_core_path)
            print(f"Continuing training from previous checkpoint: {prev_core_path}")
        else:
            args_sof.load_sfmppo = None
            args_sof.load_upn = None

        new_model_filename = f"soo_ppo_task_{i}.pth"
        new_upn_filename = f"soo_core_task_{i}.pth"
        args_sof.save_sfmppo = new_model_filename
        args_sof.save_sfm = new_upn_filename

        train_sofppo_agent(envs=cur_env)
        
        if i!=0:
            # Close the training env and start evals
            retention_log = {}
            cur_env.close()
            eval_env = gym.vector.SyncVectorEnv(
                [
                    make_env_with_wrapper(
                        env_id=env_id,
                        idx=env_idx,
                        capture_video=args_sof.capture_video,
                        run_name=args_sof.exp_name,
                        gamma=args_sof.gamma,
                        wrappers=prev_wrapper
                    )
                    for env_idx in range(args_sof.num_envs)
                ]
            )
            
            new_model_path = os.path.join(os.getcwd(), "params", "multi_task", "soo_ppo", new_model_filename)
            agent = Agent_sof(eval_env).to(args_sof.device)
            agent.load_state_dict(torch.load(new_model_path))

            print(f"\n[INFO] Evaluating after training task #{i}...")

            # Evaluate on each environment j=0..i
            for j in range(i+1):
                old_env_id = TASKS[j]["env_id"]
                old_wrappers = TASKS[j]["wrappers"]
                env_key = f"env_{j}"

                # Evaluate
                avg_return = evaluate_agent(
                    agent,
                    env_id=old_env_id,
                    wrappers=old_wrappers,
                    num_episodes=5,  # or however many you like
                    device=args_sof.device
                )

                # Store in retention_log
                if env_key not in retention_log:
                    retention_log[env_key] = []
                retention_log[env_key].append(avg_return)
                print(f" - Performance on env_{j} (task={TASKS[j]['name']}): {avg_return:.2f}")
            
            plot_retention_curves(retention_log, i)
        
        prev_wrapper = wrappers


if __name__ == "__main__":
    train_multiple_tasks_sequentially()
