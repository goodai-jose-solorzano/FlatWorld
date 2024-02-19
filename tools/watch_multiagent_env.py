import math
import time

import numpy as np
from flat_world.FlatWorldEnvironment import FlatWorldEnvironment
from flat_world.PredatorLevel import PredatorLevel
from flat_world.ref_agents.RandomAgent import RandomAgent
from flat_world.tasks.multi_agent.big_grid_config import BigGridConfig
from flat_world.tasks.multi_agent.three_agents_and_a_predator_config import ThreeAgentsAndAPredatorConfig
from timeit import default_timer as timer

env = FlatWorldEnvironment(ThreeAgentsAndAPredatorConfig(), num_agents=3, predator_level=PredatorLevel.HARD)
agent = RandomAgent()
time_limit = 120  # Seconds
step_pause = 0.1  # Seconds

if __name__ == '__main__':
    time1 = timer()
    observation = env.reset()
    while True:
        env.render()
        action = agent.get_action(env, observation)
        observation, reward, done, info = env.step(action)
        if np.all(done):
            observation = env.reset()
        time2 = timer()
        if time2 - time1 >= time_limit:
            break
        time.sleep(step_pause)

