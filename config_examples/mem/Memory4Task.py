import math
from typing import List, Tuple

from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldElement import FlatWorldElement


class Memory4Task(FlatWorldConfig):
    def __init__(self, level=10):
        super().__init__()
        self.level = level
        self.initial_agent_pos = None

    def get_grid_size(self) -> Tuple[int, int]:
        return 9, 9,

    def get_initial_agent_position(self) -> Tuple[int, int]:
        gw, gh = self.get_grid_size()
        mid_x = gw // 2
        if self.level == 0:
            x = -1
            y = -1
            wall_x1 = 2
            wall_x2 = 6
            while not self.valid_agent_pos(x, y):
                if self.random.uniform() < 0.5:
                    x = self.random.randint(0, wall_x1 + 1)
                else:
                    x = self.random.randint(wall_x2, gw)
                y = self.random.randint(0, 8)
            self.initial_agent_pos = x, y,
        elif self.level == 1:
            x = -1
            y = -1
            while not self.valid_agent_pos(x, y):
                x = self.random.randint(1, gw - 1)
                y = self.random.randint(0, 8)
            self.initial_agent_pos = x, y,
        else:
            self.initial_agent_pos = mid_x, 7,
        return self.initial_agent_pos

    def get_initial_agent_angle(self) -> float:
        if self.level <= 1:
            gw, gh = self.get_grid_size()
            mid_x = gw // 2
            ax, ay = self.initial_agent_pos
            wall_x1 = 2
            wall_x2 = 6
            if mid_x == ax:
                if ay >= 4:
                    return -math.pi / 2
                else:
                    return self.random.uniform(-math.pi * 2 / 3, -math.pi / 3)
            elif mid_x > ax > wall_x1:
                return self.random.uniform(-math.pi * 2 / 3, -math.pi / 2)
            elif ax < wall_x1:
                return self.random.uniform(math.pi / 3, 2 * math.pi / 3)
            elif wall_x2 > ax > mid_x:
                return self.random.uniform(-math.pi / 2, -math.pi / 3)
            elif ax > wall_x2:
                return self.random.uniform(math.pi / 3, 2 * math.pi / 3)
            elif ax == wall_x1:
                return math.pi
            elif ax == wall_x2:
                return 0
            else:
                raise Exception(f'Unexpected ax={ax}')
        else:
            return -math.pi / 2

    def get_min_cumulative_reward(self):
        if self.level == 0:
            return -0.5
        elif self.level <= 2:
            return -0.95
        else:
            return -0.85

    def get_elements(self, agent_x: int, agent_y: int) -> List[FlatWorldElement]:
        elements = []
        gw, gh = self.get_grid_size()
        mid_x = gw // 2
        if agent_x != mid_x:
            is_left = agent_x < mid_x
        else:
            is_left = self.random.choice([True, False])
        # Top horizontal
        for x in range(2, gw - 2):
            if x < mid_x - 1 or x > mid_x + 1:
                elements.append(FlatWorldElement.wall_brick((x, 2)))
        # Top blocker
        if self.level <= 2:
            for y in range(2, 3):
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
