import math
import numpy as np
from typing import List, Tuple
from flat_world.FlatWorldElement import FlatWorldElement
from flat_world.FlatWorldMultiAgentConfig import FlatWorldMultiAgentConfig

GRID_WIDTH = 25
GRID_HEIGHT = 25


class FoodForAllConfig(FlatWorldMultiAgentConfig):
    def get_grid_size(self) -> Tuple[int, int]:
        return GRID_WIDTH, GRID_HEIGHT,

    def get_min_cumulative_reward(self):
        return -2.0

    def get_initial_agent_angles(self, num_agents: int) -> np.ndarray:
        return self.random.uniform(-math.pi * 2, math.pi * 2, size=num_agents)

    def get_initial_agent_positions(self, num_agents: int) -> List[Tuple[int, int]]:
        grid_area = GRID_WIDTH * GRID_HEIGHT
        if num_agents > grid_area / 5:
            raise Exception('Too many agents requested!')
        sampled_pos = self.random.choice(range(grid_area), size=num_agents, replace=False)
        x, y = list(sampled_pos % GRID_WIDTH), list(sampled_pos // GRID_WIDTH)
        return list(zip(x, y))

    def get_elements_for_agents(self, initial_agent_positions: List[Tuple[int, int]]) -> List[FlatWorldElement]:
        num_agents = len(initial_agent_positions)
        grid_area = GRID_WIDTH * GRID_HEIGHT
        all_pos_set = set(range(grid_area))
        agent_pos_set = set([y * GRID_WIDTH + x for x, y in initial_agent_positions])
        avail_positions = all_pos_set - agent_pos_set
        sampled_pos = self.random.choice(list(avail_positions), size=num_agents, replace=False)
        x, y = list(sampled_pos % GRID_WIDTH), list(sampled_pos // GRID_WIDTH)
        return [FlatWorldElement.final_food((x, y)) for x, y in zip(x, y)]
