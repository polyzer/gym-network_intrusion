import ray
from ray import tune
from ray.rllib.agents import ppo
import gym
from new_gym_network_intrusion.envs.network_intrusion_env_1 import NetworkIntrusionEnv

env = NetworkIntrusionEnv


# ray.init()
# Configure the algorithm.
config = {
    "env": env,
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 5,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "tf",
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
    },
    # Set up a separate evaluation worker set for the
    # `algo.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
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
