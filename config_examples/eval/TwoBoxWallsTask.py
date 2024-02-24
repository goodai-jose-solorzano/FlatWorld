import math
from typing import List, Tuple

from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldElement import FlatWorldElement


class TwoBoxWallsTask(FlatWorldConfig):
    def get_grid_size(self) -> Tuple[int, int]:
        return 11, 9,

    def get_initial_agent_position(self) -> Tuple[int, int]:
        _, gh = self.get_grid_size()
        return 2, gh - 1,

    def get_initial_agent_angle(self) -> float:
        return self.random.uniform(0, 2 * math.pi)

    def get_elements(self, agent_x: int, agent_y: int) -> List[FlatWorldElement]:
        gw, gh = self.get_grid_size()
        elements = []
        wall_y = 4
        wall_x = 6
        # First horizontal wall
        for x in range(0, wall_x):
            elements.append(FlatWorldElement.box((x, wall_y)))
        elements.append(FlatWorldElement.translucent_brick((wall_x, wall_y)))
        # Second horizontal wall
        for x in range(wall_x + 1, gw):
            elements.append(FlatWorldElement.wall_brick((x, wall_y)))
        # Vertical wall
        for y in range(0, wall_y):
            elements.append(FlatWorldElement.box((wall_x, y)))
        # Food
        food_x = self.random.randint(wall_x + 1, gw)
        food_y = self.random.randint(0, wall_y)
        elements.append(FlatWorldElement.final_food((food_x, food_y)))
        return elements
