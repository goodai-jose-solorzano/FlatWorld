import math
import random
import numpy as np
from typing import List, Tuple
from flat_world.FlatWorldElement import FlatWorldElement
from flat_world.FlatWorldMultiAgentConfig import FlatWorldMultiAgentConfig

GRID_WIDTH = 45
GRID_HEIGHT = 45


class BigGridConfig(FlatWorldMultiAgentConfig):
    # Meant for a 100 agents, i.e. ~5% of grid cells.

    def get_grid_size(self) -> Tuple[int, int]:
        return GRID_WIDTH, GRID_HEIGHT,

    def get_min_cumulative_reward(self):
        return -3.0

    def get_initial_agent_angles(self, num_agents: int) -> np.ndarray:
        return np.random.uniform(-math.pi * 2, math.pi * 2, size=num_agents)

    def get_initial_agent_positions(self, num_agents: int) -> List[Tuple[int, int]]:
        grid_area = GRID_WIDTH * GRID_HEIGHT
        if num_agents > grid_area / 5:
            raise Exception('Too many agents requested!')
        sampled_pos = np.random.choice(range(grid_area), size=num_agents, replace=False)
        x, y = list(sampled_pos % GRID_WIDTH), list(sampled_pos // GRID_WIDTH)
        return list(zip(x, y))

    def get_elements_for_agents(self, initial_agent_positions: List[Tuple[int, int]]) -> List[FlatWorldElement]:
        g_w, g_h = self.get_grid_size()
        pos_set = set(initial_agent_positions)
        # One food per agent
        elements = []
        for _ in range(len(initial_agent_positions)):
            food_pos: Tuple[int, int] = None
            for attempt in range(100):
                bp = np.random.randint(0, g_w), np.random.randint(0, g_h)
                if bp not in pos_set:
                    food_pos = bp
                    break
            if food_pos is not None:
                food_fn = np.random.choice([FlatWorldElement.final_food, FlatWorldElement.final_food_2])
                elements.append(food_fn(food_pos))
                pos_set.add(food_pos)
        return elements

    def get_initial_predator_positions(self, initial_agent_positions: List[Tuple[int, int]],
                                       elements: List[FlatWorldElement]) -> List[Tuple[int, int]]:
        g_w, g_h = self.get_grid_size()
        pos_set = set(initial_agent_positions)
        element_positions = [e.position for e in elements]
        pos_set.update(element_positions)
        result = []
        for _ in range(len(initial_agent_positions) // 10):
            pred_pos: Tuple[int, int] = None
            for attempt in range(100):
                bp = np.random.randint(0, g_w), np.random.randint(0, g_h)
                if bp not in pos_set:
                    pred_pos = bp
                    break
            if pred_pos is not None:
                result.append(pred_pos)
                pos_set.add(pred_pos)
        return result
