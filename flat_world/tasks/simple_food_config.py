import math
from typing import List, Tuple

from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldElement import FlatWorldElement


class SimpleFoodConfig(FlatWorldConfig):
    def __init__(self):
        super().__init__()

    def get_initial_agent_position(self) -> Tuple[int, int]:
        gw, gh = self.get_grid_size()
        x = self.random.randint(0, gw)
        return x, gh - 1,

    def get_initial_agent_angle(self) -> float:
        return -math.pi / 2

    def get_elements(self, agent_x: int, agent_y: int) -> List[FlatWorldElement]:
        gw, gh = self.get_grid_size()
        mid_x, mid_y = gw // 2, gh // 2,
        elements = [
            FlatWorldElement.food((mid_x + 1, mid_y)),
            FlatWorldElement.final_food((mid_x - 2, mid_y - 2))
        ]
        return elements
