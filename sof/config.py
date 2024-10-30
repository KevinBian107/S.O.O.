from dataclasses import dataclass

@dataclass
class Args_sof:
    exp_name: str = "sofppo_halfcheetah"
    env_id: str = "HalfCheetah-v4"
    device: str = 'cpu'
    total_timesteps: int = 3000000
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = True
    seed: int = 1
    ppo_learning_rate: float = 6e-5
    upn_learning_rate: float = 8e-6
    latent_size: int = 100
    upn_hidden_layer: int = 64
    ppo_hidden_layer: int = 256
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 10
    num_minibatches: int = 32
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    upn_coef: float = 0.8
    kl_coef: float = 0.1

    # exactly how far we want distribution to be
    # what's good for suboptimal
    epsilon_k: float = 0.01
    eta_k: float = 1.0

    # when constrain_weights = 0, no constrain on MOMPO constrain
    constrain_weights: float = 0.8

    # this helps greatly for fmppo
    mix_coord: bool = False
    
    # Data need to match up, this data may be problematic
    load_upn: str = "supervised_vae_jump.pth"

    # can still use this becuase only load in PPO
    load_sfmppo: str = "sfmppo_stable.pth"

    imitation_data_path: str = None
    save_sfm: str = "sof_try.pth"
    save_sfmppo: str = "sofppo_try.pth"

    # to be set at runtime
    batch_size: int = 0 
    minibatch_size: int = 0
    iterations: int = 0

@dataclass
class Args_ppo:
    exp_name: str = "ppo_halfcheetah"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    env_id: str = "HalfCheetah-v4"
    device: str = 'cpu'
    capture_video: bool = True
    total_timesteps: int = 2000000
    learning_rate: float = 1e-5
    ppo_hidden_layer: int = 256
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 64
    update_epochs: int = 20
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    kl_coef: float = 0.1
    target_kl: float = 0.01 # the targeted KL does work well
    max_grad_norm: float = 0.5
    action_reg_coef: float = 0.0
    load_model: str = None #"ppo_stable.pth"
    save_path: str = "ppo_jump_intention.pth"

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

@dataclass
class Args_supp:
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    batch_size: int = 64
    upn_hidden_layer: int = 64
    latent_size: int = 100
    num_epochs: int = 100
    cuda: bool = True
    imitate_data_path: str = 'imitate_ppo_hard_jump_intention.npz'
    save_supp_path: str = "supervised_vae_jump.pth"

@dataclass
class Args_test:
    exp_name: str = "test"
    env_id: str = "HalfCheetah-v4"
    seed: int = 123
    capture_video: bool = True
    torch_deterministic: bool = True
    cuda: bool = True
    gamma: float = 0.99
    num_envs: int = 1
    test_episode_num: int = 200
    ppo_path: str = "ppo_hc_kl.pth"
    sof_path: str = "sfmppo_try.pth"

# initiate
args_sof = Args_sof()
args_ppo = Args_ppo()
args_supp = Args_supp()
args_test = Args_test()
