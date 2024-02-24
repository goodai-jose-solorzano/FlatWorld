import math
from typing import List, Tuple

from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldElement import FlatWorldElement


class PierceWallTask(FlatWorldConfig):
    def __init__(self, level=5):
        super().__init__()
        self.level = level

    def get_grid_size(self) -> Tuple[int, int]:
        return 11, 9,

    def get_initial_agent_position(self) -> Tuple[int, int]:
        _, gh = self.get_grid_size()
        return self.random.randint(1, 5), self.random.randint(gh - 2, gh)

    def get_initial_agent_angle(self) -> float:
        return self.random.uniform(-math.pi / 4, -math.pi * 3 / 4)

    def get_elements(self, agent_x: int, agent_y: int) -> List[FlatWorldElement]:
        gw, gh = self.get_grid_size()
        elements = []
        wall_y = 4
        tw_top_x = 6
        # Transparent part of wall
        for x in range(0, tw_top_x):
            elements.append(FlatWorldElement.translucent_brick((x, wall_y)))
        # Opaque part of wall
        bx = self.random.randint(7, 10)
        for x in range(tw_top_x, gw):
            if self.level > 1 or x != bx:
                elements.append(FlatWorldElement.wall_brick((x, wall_y)))
        # Box
        if self.level > 0:
            if self.level == 1:
                by = wall_y
            elif self.level == 2:
                by = wall_y + 1
            else:
                by = wall_y + 2
            elements.append(FlatWorldElement.box((bx, by)))
        # Food
        food_top_x = tw_top_x + 1
        avail_positions = list(range(food_top_x * wall_y))
        selected_positions = self.random.choice(avail_positions, size=3, replace=False)
        first = True
        for sp in selected_positions:
            x, y = sp % food_top_x, sp // food_top_x
            if first:
                first = False
                elements.append(FlatWorldElement.final_food((x, y)))
            else:
                elements.append(FlatWorldElement.food((x, y)))
        return elements
