import math
from typing import List, Tuple

from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldElement import FlatWorldElement


class SimpleFood2Config(FlatWorldConfig):
    def __init__(self):
        super().__init__()

    def get_grid_size(self) -> Tuple[int, int]:
        return 11, 11,

    def get_initial_agent_position(self) -> Tuple[int, int]:
        gw, gh = self.get_grid_size()
        return gw // 2, gh // 2,

    def get_initial_agent_angle(self) -> float:
        return self.random.uniform(0, 2 * math.pi)

    def get_elements(self, agent_x: int, agent_y: int) -> List[FlatWorldElement]:
        gw, gh = self.get_grid_size()
        mid_x, mid_y = gw // 2 - 1, gh // 2 - 1,
        food_x = self.random.randint(0, gw)
        food_y = self.random.randint(0, gh)
        elements = [
            FlatWorldElement.final_food((food_x, mid_y)),
        ]
        if (mid_x, food_y) != (food_x, mid_y):
            elements.append(FlatWorldElement.food((mid_x, food_y)))
        return elements
