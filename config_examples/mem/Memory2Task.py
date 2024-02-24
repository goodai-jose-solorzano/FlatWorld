import math
from typing import List, Tuple

from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldElement import FlatWorldElement


class Memory2Task(FlatWorldConfig):
    def __init__(self, level=10):
        super().__init__()
        self.level = level

    def get_grid_size(self) -> Tuple[int, int]:
        return 9, 9,

    def get_initial_agent_position(self) -> Tuple[int, int]:
        gw, gh = self.get_grid_size()
        return gw // 2, 7,

    def get_initial_agent_angle(self) -> float:
        return -math.pi / 2

    def get_min_cumulative_reward(self):
        if self.level == 0:
            return -2.0
        elif self.level == 1:
            return -1.5
        elif self.level == 2:
            return -1.5
        else:
            return -1.0

    def get_elements(self, agent_x: int, agent_y: int) -> List[FlatWorldElement]:
        elements = []
        gw, gh = self.get_grid_size()
        mid_x = gw // 2
        # Top horizontal
        for x in range(2, gw - 2):
            if x < mid_x - 1 or x > mid_x + 1:
                elements.append(FlatWorldElement.wall_brick((x, 2)))
        # Top entrance
        if self.level > 0 or self.random.uniform() < 0.5:
            elements.append(FlatWorldElement.wall_brick((mid_x - 1, 2)))
        if self.level > 0 or self.random.uniform() < 0.5:
            elements.append(FlatWorldElement.wall_brick((mid_x + 1, 2)))
        # Sides vertical and hints
        is_left = self.random.choice([True, False])
        for y in range(3, 5):
            left_brick_fn = right_brick_fn = FlatWorldElement.wall_brick
            if is_left:
                left_brick_fn = FlatWorldElement.translucent_brick
            else:
                right_brick_fn = FlatWorldElement.translucent_brick
            elements.append(left_brick_fn((2, y)))
            elements.append(right_brick_fn((gw - 3, y)))
        # Bottom horizontal
        for x in range(0, 3):
            elements.append(FlatWorldElement.wall_brick((x, 5)))
            elements.append(FlatWorldElement.wall_brick((gw - x - 1, 5)))
        # Food
        if self.level == 0:
            food_x_offset = self.random.choice([1, 2])
            food_x = mid_x - food_x_offset if is_left else mid_x + food_x_offset
            food_y = self.random.randint(0, 2)
        elif self.level == 1:
            food_x_offset = self.random.randint(1, mid_x + 1)
            food_x = mid_x - food_x_offset if is_left else mid_x + food_x_offset
            food_y = self.random.randint(0, 2)
        elif self.level == 2:
            food_x_offset = self.random.randint(mid_x - 1, mid_x + 1)
            food_x = mid_x - food_x_offset if is_left else mid_x + food_x_offset
            food_y = self.random.randint(0, 3)
        elif self.level == 3:
            food_x_offset = self.random.randint(mid_x - 1, mid_x + 1)
            food_x = mid_x - food_x_offset if is_left else mid_x + food_x_offset
            food_y = self.random.randint(1, 5)
        else:
            food_x = 0 if is_left else gw - 1
            food_y = 4
        elements.append(FlatWorldElement.final_food((food_x, food_y)))
        return elements
