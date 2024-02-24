import math
from typing import List, Tuple

from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldElement import FlatWorldElement


class HarderCylinderTask(FlatWorldConfig):
    def get_grid_size(self) -> Tuple[int, int]:
        return 11, 11,

    def get_initial_agent_position(self) -> Tuple[int, int]:
        gw, gh = self.get_grid_size()
        x = self.random.randint(0, gw)
        y = 3
        return x, y,

    def get_initial_agent_angle(self) -> float:
        return -math.pi / 2

    def get_elements(self, agent_x: int, agent_y: int) -> List[FlatWorldElement]:
        gw, gh = self.get_grid_size()
        wall_size = 5
        tw_x = self.random.randint(0, gw - wall_size)
        tw_y = self.random.randint(4, gh - 2)
        food_x = self.random.randint(tw_x + 1, tw_x + wall_size - 1)
        food_y = tw_y + 1
        elements = [FlatWorldElement.final_food_2((food_x, food_y))]
        for x in range(tw_x, tw_x + wall_size):
            elements.append(FlatWorldElement.translucent_brick((x, tw_y)))
            elements.append(FlatWorldElement.translucent_brick((x, tw_y + 2)))
        # Upper wall
        for x in range(gw):
            elements.append(FlatWorldElement.translucent_brick((x, 1)))
        # Upper food
        all_x_positions = list(range(gw))
        sampled_positions = self.random.choice(all_x_positions, 2)
        x1, x2 = tuple(sampled_positions)
        elements.append(FlatWorldElement.food((x1, 0)))
        elements.append(FlatWorldElement.final_food((x2, 0)))
        return elements
