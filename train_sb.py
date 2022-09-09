from stable_baselines3 import DQN, PPO
from new_gym_network_intrusion.envs.network_intrusion_env_1 import NetworkIntrusionEnv

env = NetworkIntrusionEnv({})
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save("ppo_network_intrusion_detection")