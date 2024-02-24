import math
from typing import List, Tuple

from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldElement import FlatWorldElement


class TransparentCylinderTask(FlatWorldConfig):
    def get_grid_size(self) -> Tuple[int, int]:
        return 11, 11,

    def get_initial_agent_position(self) -> Tuple[int, int]:
        gw, gh = self.get_grid_size()
        x = self.random.randint(0, gw)
        y = self.random.randint(gh - 3, gh)
        return x, y,

    def get_initial_agent_angle(self) -> float:
        return self.random.uniform(-3 * math.pi / 4, -math.pi / 4)

    def get_elements(self, agent_x: int, agent_y: int) -> List[FlatWorldElement]:
        gw, gh = self.get_grid_size()
        wall_size = self.random.randint(3, 6)
        tw_x = self.random.randint(1, gw - wall_size - 1)
        tw_y = self.random.randint(1, 4)
        food_x = self.random.randint(tw_x + 1, tw_x + wall_size - 1)
        food_y = self.random.randint(tw_y + 1, tw_y + 3)
        elements = [FlatWorldElement.final_food((food_x, food_y))]
        for x in range(tw_x, tw_x + wall_size):
            elements.append(FlatWorldElement.translucent_brick((x, tw_y)))
            elements.append(FlatWorldElement.translucent_brick((x, tw_y + 3)))
        return elements
