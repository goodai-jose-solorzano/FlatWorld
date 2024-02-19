import math
import unittest

import numpy as np
from numpy.random import RandomState

from flat_world.ActuatorType import ActuatorType
from flat_world.BaseEnvCharacter import ACTION_FORWARD
from flat_world.EnvState import EnvState
from flat_world.PredatorLevel import PredatorLevel
from flat_world.helpers.PrecalculatedOrientations import PrecalculatedOrientations
from flat_world.tasks.testing.box_and_blue_food_config import BoxAndBlueFoodConfig
from flat_world.tasks.testing.one_fence_brick_config import OneFenceBrickConfig
from flat_world.tasks.testing.simple_3_agent_config import Simple3AgentConfig


class TestEnvState(unittest.TestCase):
    def test_two_agent_view(self):
        config = Simple3AgentConfig()
        num_agents = 3
        num_orientations = round(2 * math.pi / config.get_rotation_step())
        obs_resolution = 20
        orientation_helper = PrecalculatedOrientations(num_orientations, obs_resolution,
                                                       config.get_peripheral_vision_range())
        random = RandomState()
        env_state = EnvState(config, random, orientation_helper, num_agents=num_agents, obs_resolution=obs_resolution,
                             default_reward=-0.01, reward_no_action=-0.01, reward_toggle=-0.01,
                             reward_predator=-1.0,
                             min_cumulative_reward=-3.0,
                             predator_level=PredatorLevel.MEDIUM,
                             actuator_type=ActuatorType.DEFAULT,
                             squeeze=False)
        for i in range(num_agents):
            agent_state = env_state.get_agent_state(i)
            assert agent_state.index == i
        bkg_color = (0, 0, 0, 255)
        observation = env_state.get_observation(bkg_color)
        last_obs = observation[2]
        assert np.max(last_obs[0]) > 100
        assert np.max(last_obs[1]) > 100
        assert np.max(last_obs[2]) > 100

    def test_not_seeing_dead_agents(self):
        config = Simple3AgentConfig()
        num_agents = 3
        num_orientations = round(2 * math.pi / config.get_rotation_step())
        obs_resolution = 20
        orientation_helper = PrecalculatedOrientations(num_orientations, obs_resolution,
                                                       config.get_peripheral_vision_range())
        random = RandomState()
        env_state = EnvState(config, random, orientation_helper, num_agents=num_agents, obs_resolution=obs_resolution,
                             default_reward=-0.01, reward_no_action=-0.01, reward_toggle=-0.01,
                             reward_predator=-1.0,
                             min_cumulative_reward=-3.0,
                             predator_level=PredatorLevel.MEDIUM,
                             actuator_type=ActuatorType.DEFAULT,
                             squeeze=False)
        env_state.get_agent_state(0).is_done = True
        env_state.get_agent_state(1).is_done = True
        bkg_color = (0, 0, 0, 255)
        observation = env_state.get_observation(bkg_color)
        assert (num_agents, 4, obs_resolution) == observation.shape
        last_obs = observation[2]
        assert np.max(last_obs[0]) < 1
        assert np.max(last_obs[1]) < 1
        assert np.max(last_obs[2]) < 1

    def test_blue_food_conversion(self):
        config = BoxAndBlueFoodConfig()
        num_agents = 1
        dr = -0.012
        num_orientations = round(2 * math.pi / config.get_rotation_step())
        obs_resolution = 20
        orientation_helper = PrecalculatedOrientations(num_orientations, obs_resolution,
                                                       config.get_peripheral_vision_range())
        random = RandomState()
        env_state = EnvState(config, random, orientation_helper, num_agents=num_agents, obs_resolution=obs_resolution,
                             default_reward=dr, reward_no_action=-0.01, reward_toggle=-0.01,
                             reward_predator=-1.0,
                             min_cumulative_reward=-3.0,
                             predator_level=PredatorLevel.MEDIUM,
                             actuator_type=ActuatorType.DEFAULT,
                             squeeze=True)
        elements = env_state.get_elements()
        assert len(elements) == 2
        reward = env_state.apply_actions(ACTION_FORWARD)
        assert math.isclose(dr, reward)
        elements = env_state.get_elements()
        assert len(elements) == 1
        assert math.isclose(1.0, elements[0].reward)

    def test_removal_of_all_entities(self):
        config = OneFenceBrickConfig()
        num_agents = 1
        death_r = -1.0
        num_orientations = round(2 * math.pi / config.get_rotation_step())
        obs_resolution = 20
        orientation_helper = PrecalculatedOrientations(num_orientations, obs_resolution,
                                                       config.get_peripheral_vision_range())
        random = RandomState()
        env_state = EnvState(config, random, orientation_helper, num_agents=num_agents, obs_resolution=obs_resolution,
                             default_reward=-0.01, reward_no_action=-0.01, reward_toggle=-0.01,
                             reward_predator=-1.0,
                             min_cumulative_reward=-3.0,
                             predator_level=PredatorLevel.MEDIUM,
                             actuator_type=ActuatorType.DEFAULT,
                             squeeze=True)
        elements = env_state.get_elements()
        assert len(elements) == 1
        reward = env_state.apply_actions(ACTION_FORWARD)
        assert math.isclose(death_r, reward)
        elements = env_state.get_elements()
        assert len(elements) == 0
        for i in range(num_agents):
            assert env_state.get_agent_state(i).is_done
        bkg_color = (0, 0, 0, 255)
        observation = env_state.get_observation(bkg_color)
        assert (4, obs_resolution) == observation.shape
