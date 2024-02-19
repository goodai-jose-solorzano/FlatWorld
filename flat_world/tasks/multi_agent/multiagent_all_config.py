import math
import numpy as np
from typing import List, Tuple, Union
from flat_world.FlatWorldElement import FlatWorldElement
from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldMultiAgentConfig import FlatWorldMultiAgentConfig


class MultiAgentAllConfig(FlatWorldMultiAgentConfig):
    def get_initial_agent_angles(self, num_agents: int) -> Union[List[float], np.ndarray]:
        return self.random.uniform(0, math.pi * 2, size=num_agents)

    def get_initial_agent_positions(self, num_agents: int) -> List[Tuple[int, int]]:
        g_w, g_h = self.get_grid_size()
        positions = self.random.choice(range(g_w * g_h), size=num_agents, replace=False)
        return [(p % g_w, p // g_w) for p in positions]

    def get_elements_for_agents(self, initial_agent_positions: List[Tuple[int, int]]) -> List[FlatWorldElement]:
        g_w, g_h = self.get_grid_size()
        pos_set = set(initial_agent_positions)
        elements = []
        for element_fn in [FlatWorldElement.food, FlatWorldElement.final_food, FlatWorldElement.final_food_2,
                           FlatWorldElement.box, FlatWorldElement.fence_brick, FlatWorldElement.translucent_brick]:
            box_pos: Tuple[int, int] = None
            for attempt in range(100):
                bp = self.random.randint(1, g_w - 1), self.random.randint(1, g_h - 1)
                if bp not in pos_set:
                    box_pos = bp
                    break
            if box_pos is not None:
                elements.append(element_fn(box_pos))
                pos_set.add(box_pos)
        return elements
