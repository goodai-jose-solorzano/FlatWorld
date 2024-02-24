import math
from typing import List, Tuple

from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldElement import FlatWorldElement


class Memory5Task(FlatWorldConfig):
    def __init__(self, level=10, forced_right=None):
        super().__init__()
        self.level = level
        self.forced_right = forced_right

    def get_grid_size(self) -> Tuple[int, int]:
        return 9, 9,

    def get_initial_agent_position(self) -> Tuple[int, int]:
        gw, gh = self.get_grid_size()
        mid_x = gw // 2
        ay = 7 if self.level > 0 else self.random.randint(5, 8)
        return mid_x, ay,

    def get_initial_agent_angle(self) -> float:
        return -math.pi / 2

    def get_min_cumulative_reward(self):
        if self.level == 0:
            return -0.80
        else:
            return -0.95

    def get_elements(self, agent_x: int, agent_y: int) -> List[FlatWorldElement]:
        elements = []
        gw, gh = self.get_grid_size()
        mid_x = gw // 2
        if self.forced_right is None:
            is_left = self.random.choice([True, False])
        else:
            is_left = not self.forced_right
        # Top horizontal
        for x in range(2, gw - 2):
            if x < mid_x - 1 or x > mid_x + 1:
                elements.append(FlatWorldElement.wall_brick((x, 2)))
        # Top blocker
        if self.level <= 2:
            if self.level <= 1 or self.random.uniform() < 0.75:
                for y in range(0, 3):
                    elements.append(FlatWorldElement.wall_brick((mid_x, y)))
        # Top entrance
        if self.level > 2:
            if self.level > 3 or self.random.uniform() < 0.5:
                elements.append(FlatWorldElement.wall_brick((mid_x - 1, 2)))
            if self.level > 3 or self.random.uniform() < 0.5:
                elements.append(FlatWorldElement.wall_brick((mid_x + 1, 2)))
        # Sides vertical and hints
        for y in range(3, 5):
            left_brick_fn = right_brick_fn = FlatWorldElement.wall_brick
            if is_left:
                right_brick_fn = FlatWorldElement.fence_brick
            else:
                left_brick_fn = FlatWorldElement.fence_brick
            elements.append(left_brick_fn((2, y)))
            elements.append(right_brick_fn((gw - 3, y)))
        # Bottom horizontal
        for x in range(0, 3):
            elements.append(FlatWorldElement.wall_brick((x, 5)))
            elements.append(FlatWorldElement.wall_brick((gw - x - 1, 5)))
        # Food
        if self.level == 0:
            wall_x1 = 2
            wall_x2 = gw - 3
            offset = self.random.choice([0, 1])
            food_x = wall_x1 - offset if is_left else wall_x2 + offset
            food_y = 0
        elif self.level == 1:
            wall_x1 = 2
            wall_x2 = gw - 3
            offset = self.random.choice([1, 2])
            food_x = wall_x1 - offset if is_left else wall_x2 + offset
            food_y = 1
        elif self.level == 2:
            wall_x1 = 2
            wall_x2 = gw - 3
            food_x = wall_x1 - 2 if is_left else wall_x2 + 2
            food_y = 2
        else:
            food_x = 0 if is_left else gw - 1
            food_y = 4
        elements.append(FlatWorldElement.final_food((food_x, food_y)))
        return elements

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
