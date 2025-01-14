import os
import gymnasium as gym
import numpy as np

from train_soo import train_sofppo_agent
from env_wrappers import (
    TargetVelocityWrapper,
    JumpRewardWrapper,
)
from environments import make_env_with_wrapper
from config import args_sof

TASKS = [
    {
        "env_id": "HalfCheetah-v4",
        "name": "Target Velocity = 1.5",
        "wrappers": [lambda env: TargetVelocityWrapper(env, target_velocity=1.5)],
    },
    {
        "env_id": "HalfCheetah-v4",
        "name": "Target Jump = 1.5",
        "wrappers": [lambda env: JumpRewardWrapper(env, jump_target_height=1.5)],
    },
    {
        "env_id": "HalfCheetah-v4",
        "name": "Target Velocity = 2.5",
        "wrappers": [lambda env: TargetVelocityWrapper(env, target_velocity=2.5)],
    },
]

def train_multiple_tasks_sequentially():
    """
    Trains on multiple tasks (environments) in sequence,
    loading the previous model checkpoint and continuing training for the next task.
    This version does NOT evaluate or log performance—only training.
    """

    multi_task_save_dir = os.path.join(os.getcwd(), "sof", "params", "multi_task")
    os.makedirs(multi_task_save_dir, exist_ok=True)

    for i, task_info in enumerate(TASKS):
        env_id = task_info["env_id"]
        task_name = task_info["name"]
        wrappers = task_info["wrappers"]

        # Build a vectorized environment for the current task.
        # We assume your `make_env` can handle custom env IDs or wrapper logic.
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

        # If this is NOT the first task, load the previous task's model
        if i > 0:
            prev_model_path = os.path.join(multi_task_save_dir, f"task_{i-1}.pth")
            args_sof.load_sfmppo = os.path.basename(prev_model_path)
            print(f"Continuing training from previous checkpoint: {prev_model_path}")
        else:
            # First task: start from scratch
            args_sof.load_sfmppo = None

        new_model_filename = f"task_{i}.pth"
        args_sof.save_sfmppo = new_model_filename

        train_sofppo_agent(envs=cur_env)

if __name__ == "__main__":
    train_multiple_tasks_sequentially()
