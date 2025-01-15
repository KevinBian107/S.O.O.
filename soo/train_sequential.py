import os
import gymnasium as gym
import numpy as np

from train_soo import train_sofppo_agent
from env.env_wrappers import (
    TargetVelocityWrapper,
    JumpRewardWrapper,
    FlipRewardWrapper
)
from env.environments import make_env_with_wrapper
from config import args_sof

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

def train_multiple_tasks_sequentially():
    multi_task_save_dir = os.path.join(os.getcwd(), "params", "multi_task")
    os.makedirs(multi_task_save_dir, exist_ok=True)

    for i, task_info in enumerate(TASKS):
        env_id = task_info["env_id"]
        task_name = task_info["name"]
        wrappers = task_info["wrappers"]

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


if __name__ == "__main__":
    train_multiple_tasks_sequentially()
