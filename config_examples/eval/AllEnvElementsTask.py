import math

import numpy as np
from typing import List, Tuple, Union
from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldElement import FlatWorldElement

GRID_SIZE = 10
ENV_ELEMENTS = [FlatWorldElement.food, FlatWorldElement.final_food, FlatWorldElement.final_food_2,
                FlatWorldElement.box, FlatWorldElement.translucent_brick, FlatWorldElement.fence_brick]


class AllEnvElementsTask(FlatWorldConfig):
    def get_grid_size(self) -> Tuple[int, int]:
        return GRID_SIZE, GRID_SIZE,

    def get_initial_agent_position(self) -> Tuple[int, int]:
        return self.random.randint(0, GRID_SIZE), self.random.randint(0, GRID_SIZE),

    def get_initial_agent_angle(self) -> float:
        return self.random.uniform() * math.pi * 2

    def get_elements(self, agent_x: int, agent_y: int) -> List[object]:
        elements = []
        x1 = max(0, agent_x - 2)
        y1 = max(0, agent_y - 2)
        x2 = min(GRID_SIZE, agent_x + 3)
        y2 = min(GRID_SIZE, agent_y + 3)
        sub_grid_pos = [(c, r) for r in range(y1, y2) for c in range(x1, x2)]
        sub_grid_pos = [(c, r) for c, r in sub_grid_pos if abs(c - agent_x) >= 1 or abs(r - agent_y) >= 1]
        assert len(sub_grid_pos) >= 5
        self.random.shuffle(sub_grid_pos)
        for i, env_element_fn in enumerate(ENV_ELEMENTS):
            elements.append(env_element_fn(sub_grid_pos[i]))
        return elements
