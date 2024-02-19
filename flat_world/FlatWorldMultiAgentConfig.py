import abc
from typing import Tuple, Union, List

import numpy as np

from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldElement import FlatWorldElement


class FlatWorldMultiAgentConfig(FlatWorldConfig):
    def get_initial_agent_position(self) -> Tuple[int, int]:
        raise NotImplemented('unused')

    def get_initial_agent_angle(self) -> float:
        raise NotImplemented('unused')

    def get_elements(self, agent_x: int, agent_y: int) -> List[FlatWorldElement]:
        raise NotImplemented('unused')

    @abc.abstractmethod
    def get_initial_agent_angles(self, num_agents: int) -> Union[List[float], np.ndarray]:
        raise NotImplemented('override')

    @abc.abstractmethod
    def get_initial_agent_positions(self, num_agents: int) -> List[Tuple[int, int]]:
        raise NotImplemented('override')

    @abc.abstractmethod
    def get_elements_for_agents(self, initial_agent_positions: List[Tuple[int, int]]) -> List[FlatWorldElement]:
        raise NotImplemented('override')
