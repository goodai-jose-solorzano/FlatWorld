import unittest

import numpy as np

from flat_world.FlatWorldEnvironment import FlatWorldEnvironment
from flat_world.tasks.multi_agent.multiagent_all_config import MultiAgentAllConfig
from flat_world.tasks.starter_2_config import Starter2Config
from flat_world.helpers.ObservationScaler import ObservationScaler


class TestObservationScaler(unittest.TestCase):
    def test_scaler(self):
        scaler = ObservationScaler()
        env = FlatWorldEnvironment(Starter2Config())
        scaler.fit(env, 2000)
        observation = env.reset()
        obs_shape = observation.shape
        num_channels = obs_shape[0]
        std_obs = scaler.transform(observation)
        obs_lists = [[list(std_obs[c])] for c in range(num_channels)]
        for s in range(1000):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                observation = env.reset()
            std_obs = scaler.transform(observation)
            for c in range(num_channels):
                obs_lists[c].append(list(std_obs[c]))
        for c in range(num_channels):
            obs_list = obs_lists[c]
            obs_mean = np.mean(obs_list)
            obs_std = np.std(obs_list)
            assert abs(obs_mean) < 0.5
            assert abs(obs_std - 1.0) < 1.0

    def test_scaler_multiagent(self):
        scaler = ObservationScaler()
        num_agents = 5
        env = FlatWorldEnvironment(MultiAgentAllConfig(), num_agents=num_agents, obs_resolution=20)
        scaler.fit(env, 1000)
        observation = env.reset()
        obs_shape = observation.shape
        assert obs_shape == (5, 4, 20)
        num_channels = obs_shape[1]
        std_obs = scaler.transform(observation)
        obs_lists = [[] for c in range(num_channels)]
        for c in range(num_channels):
            obs_lists[c].extend(std_obs[:, c].flatten())
        for s in range(1000):
            actions = [env.action_space.sample() for _ in range(num_agents)]
            observation, _, done, info = env.step(actions)
            num_valid = num_agents
            if np.all(done):
                observation = env.reset()
                valid_obs = observation
            else:
                valid_obs = observation[~done]
                num_valid = sum(~done)
            std_obs = scaler.transform(valid_obs)
            assert std_obs.shape == (num_valid, 4, 20), f'std_obs shape={std_obs.shape}'
            for c in range(num_channels):
                obs_lists[c].extend(std_obs[:, c].flatten())
        for c in range(num_channels):
            obs_list = obs_lists[c]
            obs_mean = np.mean(obs_list)
            obs_std = np.std(obs_list)
            assert abs(obs_mean) < 0.5
            assert abs(obs_std - 1.0) < 1.0

