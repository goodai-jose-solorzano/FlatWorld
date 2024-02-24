import math
from typing import List, Tuple

import numpy as np
from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldElement import FlatWorldElement


class Memory7Task(FlatWorldConfig):
    def __init__(self, level=10, forced_right=None):
        super().__init__()
        self.level = level
        self.forced_right = forced_right

    def get_grid_size(self) -> Tuple[int, int]:
        return 9, 9,

    def get_initial_agent_position(self) -> Tuple[int, int]:
        gw, gh = self.get_grid_size()
        mid_x = gw // 2
        return mid_x, 5,

    def get_initial_agent_angle(self) -> float:
        return -math.pi / 2

    def get_min_cumulative_reward(self):
        if self.level == 0:
            return -0.80
        elif self.level == 1:
            return -0.90
        else:
            return -1.00

    def get_elements(self, agent_x: int, agent_y: int) -> List[FlatWorldElement]:
        permanent_elements = []
        bottom_elements = []
        side_elements = []
        top_elements = []
        entrance_elements = []
        gw, gh = self.get_grid_size()
        mid_x = gw // 2
        if self.forced_right is None:
            is_left = self.random.choice([True, False])
        else:
            is_left = not self.forced_right
        # Top horizontal
        for x in range(2, gw - 2):
            if x < mid_x - 1 or x > mid_x + 1:
                top_elements.append(FlatWorldElement.wall_brick((x, 2)))
        # Top entrance
        entrance_elements.append(FlatWorldElement.wall_brick((mid_x - 1, 2)))
        entrance_elements.append(FlatWorldElement.wall_brick((mid_x + 1, 2)))
        # Side with no hint:
        hint_x = gw - 4 if is_left else 3
        nhs_x = 3 if is_left else gw - 4
        for y in range(3, 5):
            side_elements.append(FlatWorldElement.wall_brick((2, y)))
            side_elements.append(FlatWorldElement.wall_brick((gw - 3, y)))
            side_elements.append(FlatWorldElement.wall_brick((nhs_x, y)))
        for y in range(3, 5):
            permanent_elements.append(FlatWorldElement.translucent_brick((hint_x, y)))
        # Bottom horizontal
        bottom_elements.append(FlatWorldElement.wall_brick((mid_x, 6)))
        for y in range(5, gh):
            bottom_elements.append(FlatWorldElement.wall_brick((mid_x - 1, y)))
            bottom_elements.append(FlatWorldElement.wall_brick((mid_x + 1, y)))
        # Food
        if self.level == 0:
            food_x = self.random.randint(0, mid_x) if is_left else self.random.randint(mid_x + 1, gw)
            food_y = self.random.randint(0, 2)
        elif self.level == 1:
            food_x = self.random.randint(0, 2) if is_left else self.random.randint(gw - 2, gw)
            food_y = self.random.randint(1, 5)
        else:
            food_x = self.random.randint(0, 3) if is_left else self.random.randint(gw - 3, gw)
            food_y = self.random.randint(6, gh)
        permanent_elements.append(FlatWorldElement.final_food((food_x, food_y)))
        selected_elements = bottom_elements + side_elements + top_elements + entrance_elements
        return permanent_elements + selected_elements

    def valid_agent_pos(self, ax: int, ay: int):
        gw, gh = self.get_grid_size()
        mid_x = gw // 2
        wall_x1 = 2
        wall_x2 = 6
        if ax == mid_x:
            return 7 >= ay >= 5
        elif 0 <= ax < wall_x1 or 0 <= ax > wall_x2:
            return 0 <= ay <= 3
        elif ax == wall_x1 or ax == wall_x2:
            return 0 <= ay <= 1
        elif wall_x2 > ax > wall_x1:
            return 0 <= ay < 5
        else:
            return False
