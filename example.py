import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('highway-v0', render_mode='rgb_array')
env.reset()

for _ in range(1000):
    action = env.action_type.actions_indexes['IDLE']
    obs, reward, done, _, info = env.step(action)
    env.render()

# plt.imshow(env.render())
# plt.show()