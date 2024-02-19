import math
import time

import gym.wrappers
import numpy as np
from flat_world.FlatWorldEnvironment import FlatWorldEnvironment
from flat_world.PredatorLevel import PredatorLevel
from flat_world.ref_agents.RandomAgent import RandomAgent
from flat_world.tasks.multi_agent.big_grid_config import BigGridConfig
from flat_world.tasks.multi_agent.three_agents_and_a_predator_config import ThreeAgentsAndAPredatorConfig
from timeit import default_timer as timer

from flat_world.tasks.starter_1_config import Starter1Config
from flat_world.tasks.starter_2_config import Starter2Config

env = FlatWorldEnvironment(Starter2Config(), num_agents=1)
env = gym.wrappers.Monitor(env, '/data/flatworld-videos/', force=True)
agent = RandomAgent()
num_episodes = 10

if __name__ == '__main__':
    try:
        for episode in range(num_episodes):
            observation = env.reset()
            done = False
            while not done:
                action = agent.get_action(env, observation)
                observation, reward, done, info = env.step(action)
    finally:
        env.close()
