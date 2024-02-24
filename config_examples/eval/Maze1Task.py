import math
from typing import List, Tuple

from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldElement import FlatWorldElement


class Maze1Task(FlatWorldConfig):
    def get_grid_size(self) -> Tuple[int, int]:
        return 11, 11,

    def get_min_cumulative_reward(self):
        return -2.0

    def get_initial_agent_position(self) -> Tuple[int, int]:
        _, gh = self.get_grid_size()
        return 1, gh - 1,

    def get_initial_agent_angle(self) -> float:
        return -math.pi / 2

    def get_elements(self, agent_x: int, agent_y: int) -> List[FlatWorldElement]:
        elements = []
        gw, gh = self.get_grid_size()
        # First wall
        for y in range(2, gh):
            elements.append(FlatWorldElement.wall_brick((3, y)))
        # Second wall
        for y in range(0, gh - 2):
            elements.append(FlatWorldElement.wall_brick((7, y)))
        # Food
        elements.append(FlatWorldElement.final_food((9, 0)))
        return elements
