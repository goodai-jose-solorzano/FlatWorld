import math
import numpy as np
from typing import List, Tuple, Union
from flat_world.FlatWorldElement import FlatWorldElement
from flat_world.FlatWorldConfig import FlatWorldConfig

GRID_SIZE = 10


class Starter2Config(FlatWorldConfig):
    def get_grid_size(self) -> Tuple[int, int]:
        return GRID_SIZE, GRID_SIZE,

    def get_initial_agent_position(self) -> Tuple[int, int]:
        return self.random.randint(0, GRID_SIZE), self.random.randint(0, GRID_SIZE),

    def get_initial_agent_angle(self) -> float:
        return self.random.uniform() * math.pi * 2

    def get_elements(self, agent_x: int, agent_y: int) -> List[object]:
        elements = []
        x1 = max(0, agent_x - 3)
        y1 = max(0, agent_y - 3)
        x2 = min(GRID_SIZE, agent_x + 4)
        y2 = min(GRID_SIZE, agent_y + 4)
        sub_grid_pos = [(c, r) for r in range(y1, y2) for c in range(x1, x2)]
        sub_grid_pos = [(c, r) for c, r in sub_grid_pos if abs(c - agent_x) >= 1 or abs(r - agent_y) >= 1]
        assert len(sub_grid_pos) >= 7
        self.random.shuffle(sub_grid_pos)
        elements.append(FlatWorldElement.food(sub_grid_pos[0]))
        elements.append(FlatWorldElement.final_food(sub_grid_pos[1]))
        elements.append(FlatWorldElement.final_food_2(sub_grid_pos[2]))
        elements.append(FlatWorldElement.translucent_brick(sub_grid_pos[3]))
        elements.append(FlatWorldElement.fence_brick(sub_grid_pos[4]))
        elements.append(FlatWorldElement.box(sub_grid_pos[5]))
        elements.append(FlatWorldElement.wall_brick(sub_grid_pos[6]))
        return elements
