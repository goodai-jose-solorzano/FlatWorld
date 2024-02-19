import unittest

import numpy as np

from flat_world.FlatWorldEnvironment import FlatWorldEnvironment
from flat_world.tasks.memory_config import MemoryConfig
from flat_world.tasks.blank_config import BlankConfig
from flat_world.tasks.multi_agent.food_for_all_config import FoodForAllConfig
from flat_world.tasks.starter_1_config import Starter1Config


class TestFlatWorldEnvironment(unittest.TestCase):
    def test_depth(self):
        env = FlatWorldEnvironment(BlankConfig())
        env.reset()
        agent_state = env.env_state.get_agent_state(0)
        agent_state.position = np.array([0.5, 0.5])
        agent_state.actuator.orientation_index = 2
        bkg_color = (0, 0, 0, 255)
        observation = env.env_state.get_observation(bkg_color)
        depths = observation[3]
        assert np.all(depths <= 255.0)
        assert np.all(depths >= 0.0)
        assert np.all(depths[0:-3] >= 150)
        assert np.all(depths[-1] <= depths[:-3])
        assert np.all(depths[-2] <= depths[:-3])

    def test_multiagent_1(self):
        env = FlatWorldEnvironment(FoodForAllConfig(), num_agents=7)
        obs = env.reset()
        assert obs.shape == (7, 4, 20)
        actions = [4, 3, 2, 1, 3, 2, 1]
        obs, reward, done, info = env.step(actions)
        assert obs.shape == (7, 4, 20)
        assert len(reward) == 7
        assert len(done) == 7
        assert len(info) == 7

    def test_squeeze(self):
        env = FlatWorldEnvironment(Starter1Config())
        obs = env.reset()
        assert obs.shape == (4, 20)
        action = 1
        obs, reward, done, info = env.step(action)
        assert obs.shape == (4, 20)
        assert type(reward) == float
        assert type(done) == bool

    def test_no_squeeze(self):
        env = FlatWorldEnvironment(Starter1Config(), squeeze=False)
        obs = env.reset()
        assert obs.shape == (1, 4, 20)
        action = [1]
        obs, reward, done, info = env.step(action)
        assert obs.shape == (1, 4, 20)
        assert len(reward) == 1
        assert len(done) == 1
        assert len(info) == 1

    def test_memory_task_obs(self):
        env = FlatWorldEnvironment(MemoryConfig(level=0), squeeze=False)
        env.seed(1)
        obs = env.reset()
        assert obs.shape == (1, 4, 20)
        assert np.all(obs >= -0.01)
        assert np.all(obs <= 255.01)
