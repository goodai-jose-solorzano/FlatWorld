import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
import numpy as np
from numpy.random import RandomState

from flat_world.FlatWorldElement import FlatWorldElement

PVR = np.deg2rad(110)


class FlatWorldConfig(ABC):
    def __init__(self):
        self.random = RandomState()

    def seed(self, seed: Union[float, int]):
        self.random.seed(seed)

    def get_background_color(self):
        return 0, 0, 0, 255,

    def get_grid_size(self) -> Tuple[int, int]:
        return 10, 10,

    def get_peripheral_vision_range(self):
        return PVR

    def get_rotation_step(self):
        return math.pi / 6

    def get_movement_step(self):
        return 1.0 / 3.0

    def get_min_cumulative_reward(self):
        return -1.0

    def get_initial_predator_positions(self, initial_agent_positions: List[Tuple[int, int]],
                                       elements: List[FlatWorldElement]) -> List[Tuple[int, int]]:
        return []

    @abstractmethod
    def get_initial_agent_position(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def get_initial_agent_angle(self) -> float:
        pass

    def get_initial_agent_angles(self, num_agents: int) -> Union[List[float], np.ndarray]:
        assert num_agents == 1, \
            'get_initial_agent_angles() musts be overridden in multi-agent configurations, ' + \
            'or extend FlatWorldMultiAgentConfig'
        return [self.get_initial_agent_angle()]

    def get_initial_agent_positions(self, num_agents: int) -> List[Tuple[int, int]]:
        assert num_agents == 1, \
            'get_initial_agent_positions() musts be overridden in multi-agent configurations, ' + \
            'or extend FlatWorldMultiAgentConfig'
        return [self.get_initial_agent_position()]

    @abstractmethod
    def get_elements(self, agent_x: int, agent_y: int) -> List[FlatWorldElement]:
        '''
        :param agent_x: Initial grid column of the agent.
        :param agent_y: Initial grid row of the agent.
        :return: New FlatWorldElement objects. (They should not be reused.)
        '''
        pass

    def get_elements_for_agents(self, initial_agent_positions: List[Tuple[int, int]]) -> List[FlatWorldElement]:
        '''
        It should return a list of new elements. (They should not be reused.)
        :param initial_agent_positions: Initial agent grid positions.
        :return: A list of objects of type FlatWorldElement
        '''
        assert len(initial_agent_positions) == 1, \
            'get_elements_for_agents() musts be overridden in multi-agent configurations, ' + \
            'or extend FlatWorldMultiAgentConfig'
        x, y = initial_agent_positions[0]
        return self.get_elements(x, y)
