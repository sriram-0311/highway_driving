sys.path.insert(0, '/content/highway-env/scripts/')
from utils import record_videos, show_videos

# Environment
import gym
import highway_env

from tqdm.notebook import trange

# Agent
from stable_baselines3 import DQN



# load saved model
model = DQN.load("intersection_dqn")

env = gym.make('highway-fast-v0', render_mode='rgb_array')
env = record_videos(env)
for episode in trange(3, desc='Test episodes'):
    (obs, info), done = env.reset(), False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
env.close()
show_videos()