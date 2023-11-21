import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
import sys
from tqdm.notebook import trange
sys.path.insert(0, '/home/ramesh.anu/HighwayEnv/scripts/')
# from utils import record_videos, show_videos

model = DQN('MlpPolicy', 'intersection-v0',
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction=0.7,
                verbose=1,
                tensorboard_log='highway_dqn/')
model.learn(int(2e4))

# save model
model.save("intersection_dqn")