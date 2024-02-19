import math
from typing import Union, List, Tuple

import numpy as np

from flat_world.FlatWorldElement import FlatWorldElement
from flat_world.FlatWorldMultiAgentConfig import FlatWorldMultiAgentConfig


class Simple3AgentConfig(FlatWorldMultiAgentConfig):
    # For unit testing

    def get_min_cumulative_reward(self):
        return -3.0

    def get_initial_agent_angles(self, num_agents: int) -> Union[List[float], np.ndarray]:
        assert num_agents == 3, 'Config expects 3 agents!'
        return [0, 0, -math.pi / 2]

    def get_initial_agent_positions(self, num_agents: int) -> List[Tuple[int, int]]:
        assert num_agents == 3, 'Config expects 3 agents!'
        return [
            (2, 1),
            (7, 1),
            (5, 9),
        ]

    def get_elements_for_agents(self, initial_agent_positions: List[Tuple[int, int]]) -> List[FlatWorldElement]:
        return []
