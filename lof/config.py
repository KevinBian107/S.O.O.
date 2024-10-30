from dataclasses import dataclass

@dataclass
class Args:
    exp_name: str = "sofppo_halfcheetah"
    env_id: str = "HalfCheetah-v4"
    device: str = 'cpu'
    total_timesteps: int = 1000000
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = True
    seed: int = 1
    ppo_learning_rate: float = 1e-5
    upn_learning_rate: float = 8e-6 # lower learning rate
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
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    upn_coef: float = 0.8
    kl_coef: float = 0.3

    # exactly how far we want distribution to be
    # what's good for suboptimal
    epsilon_k: float = 0.01
    eta_k: float = 1.0

    # when constrain_weights = 0, no constrain on MOMPO constrain
    constrain_weights: float = 0.8

    # this helps greatly
    mix_coord: bool = False
    
    # Data need to match up, this data may be problematic
    load_upn: str = "supp/supervised_vae_jump.pth"
    load_sfmppo: str = "sfmppo/sfmppo_stable.pth" # can still use this becuase only load in PPO

    imitation_data_path: str= None #"imitation_data_ppo_new.npz"
    save_sfm: str = "sfm/sfm_try.pth"
    save_sfmppo: str = "sfmppo/sfmppo_try.pth"

    # to be set at runtime
    batch_size: int = 0 
    minibatch_size: int = 0
    iterations: int = 0

args = Args()