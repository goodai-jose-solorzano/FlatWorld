import math
import random
import numpy as np
from typing import List, Tuple
from flat_world.FlatWorldElement import FlatWorldElement
from flat_world.FlatWorldMultiAgentConfig import FlatWorldMultiAgentConfig


class BigArenaConfig(FlatWorldMultiAgentConfig):
    # Meant for a 50 agents, i.e. ~4% of grid cells.
    def __init__(self, grid_width=35, grid_height=35):
        super().__init__()
        self.grid_width = grid_width
        self.grid_height = grid_height

    def get_grid_size(self) -> Tuple[int, int]:
        return self.grid_width, self.grid_height,

    def get_min_cumulative_reward(self):
        return -1.0

    def get_initial_agent_angles(self, num_agents: int) -> np.ndarray:
        return np.random.uniform(-math.pi * 2, math.pi * 2, size=num_agents)

    def get_reduced_grid_size(self, num_agents: int):
        expected_grid_area = num_agents / 0.04
        g_w, g_h = self.get_grid_size()
        grid_area = g_w * g_h
        if expected_grid_area > grid_area * 1.1:
            raise Exception(f'Too many agents requested: {num_agents}; would require grid area {expected_grid_area}')
        expected_side_len = int(math.sqrt(expected_grid_area))
        rgw = min(g_w, expected_side_len)
        rgh = min(g_h, expected_side_len)
        return rgw, rgh,

    def get_initial_agent_positions(self, num_agents: int) -> List[Tuple[int, int]]:
        rgw, rgh = self.get_reduced_grid_size(num_agents)
        used_grid_area = rgw * rgh
        sampled_pos = np.random.choice(range(used_grid_area), size=num_agents, replace=False)
        x, y = list(sampled_pos % rgw), list(sampled_pos // rgh)
        return list(zip(x, y))

    def get_elements_for_agents(self, initial_agent_positions: List[Tuple[int, int]]) -> List[FlatWorldElement]:
        num_agents = len(initial_agent_positions)
        g_w, g_h = self.get_reduced_grid_size(num_agents)
        pos_set = set(initial_agent_positions)
        # One food per agent
        elements = []
        num_extra_foods = sum(self.random.uniform(size=num_agents) <= 0.2)
        for idx in range(num_agents + num_extra_foods):
            food_pos: Tuple[int, int] = None
            for attempt in range(100):
                bp = np.random.randint(0, g_w), np.random.randint(0, g_h)
                if bp not in pos_set:
                    food_pos = bp
                    break
            if food_pos is not None:
                if idx < num_agents:
                    # Blue/greed food per agent
                    food_fn = np.random.choice([FlatWorldElement.final_food, FlatWorldElement.final_food_2])
                else:
                    # Extra yellow foods
                    food_fn = FlatWorldElement.food
                elements.append(food_fn(food_pos))
                pos_set.add(food_pos)
        return elements

    def get_initial_predator_positions(self, initial_agent_positions: List[Tuple[int, int]],
                                       elements: List[FlatWorldElement]) -> List[Tuple[int, int]]:
        return []
