import math
import numpy as np
from typing import Union, List, Tuple
from flat_world.FlatWorldElement import FlatWorldElement
from flat_world.FlatWorldMultiAgentConfig import FlatWorldMultiAgentConfig


class ThreeAgentsAndAPredatorConfig(FlatWorldMultiAgentConfig):
    def get_min_cumulative_reward(self):
        return -3.0

    def get_initial_agent_angles(self, num_agents: int) -> Union[List[float], np.ndarray]:
        return self.random.uniform(0, 2 * math.pi, size=num_agents)

    def get_initial_agent_positions(self, num_agents: int) -> List[Tuple[int, int]]:
        assert num_agents == 3, 'Config expects 3 agents!'
        g_w, g_h = self.get_grid_size()
        return [
            (0, 0),
            (0, g_h - 1),
            (g_w - 1, 0),
        ]

    def get_elements_for_agents(self, initial_agent_positions: List[Tuple[int, int]]) -> List[FlatWorldElement]:
        return [
            FlatWorldElement.box((4, 3)),
            FlatWorldElement.fence_brick((6, 5)),
        ]

    def get_initial_predator_positions(self, initial_agent_positions: List[Tuple[int, int]],
                                       elements: List[FlatWorldElement]) -> List[Tuple[int, int]]:
        g_w, g_h = self.get_grid_size()
        return [
            (g_w - 1, g_h - 1)
        ]
