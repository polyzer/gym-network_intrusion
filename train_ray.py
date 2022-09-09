import ray
from ray import tune
from ray.rllib.agents import ppo
import gym
from new_gym_network_intrusion.envs.network_intrusion_env_1 import NetworkIntrusionEnv

env = NetworkIntrusionEnv


# ray.init()
# Configure the algorithm.
num_workers = 10
rollout = 1000
config = {
    "env": env,
    "num_workers": num_workers,
    # "num_gpus": 1.0,
    "rollout_fragment_length": rollout,
    "train_batch_size": num_workers*rollout,
    "framework": "torch",
    "model": {
        "fcnet_hiddens": [128, 128],
        "fcnet_activation": "relu",
    },
}

stop_dict = {
    "training_iteration": 1000
}

results = tune.run(
    ppo.PPOTrainer,
    config = config,
    local_dir = "./run/PPO/",
    stop = stop_dict,
    checkpoint_at_end = True,
    checkpoint_freq = 1,
    verbose = 1,
    log_to_file=True,
)
