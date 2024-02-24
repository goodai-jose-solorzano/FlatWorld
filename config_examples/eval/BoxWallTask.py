import math
from typing import List, Tuple

from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldElement import FlatWorldElement


class BoxWallTask(FlatWorldConfig):
    def get_grid_size(self) -> Tuple[int, int]:
        return 9, 9,

    def get_initial_agent_position(self) -> Tuple[int, int]:
        _, gh = self.get_grid_size()
        return 2, gh - 1,

    def get_initial_agent_angle(self) -> float:
        return math.pi / 2

    def get_elements(self, agent_x: int, agent_y: int) -> List[FlatWorldElement]:
        gw, gh = self.get_grid_size()
        elements = []
        # Wall
        wall_y = 4
        for x in range(0, gw):
            elements.append(FlatWorldElement.box((x, wall_y)))
        # Food
        food_x = self.random.randint(0, gw)
        food_y = self.random.randint(0, wall_y)
        elements.append(FlatWorldElement.final_food((food_x, food_y)))
        return elements
