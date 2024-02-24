import math
from typing import List, Tuple

import numpy as np
from flat_world.FlatWorldConfig import FlatWorldConfig
from flat_world.FlatWorldElement import FlatWorldElement


class Memory6Task(FlatWorldConfig):
    def __init__(self, level=10, forced_right=None):
        super().__init__()
        self.level = level
        self.forced_right = forced_right

    def get_grid_size(self) -> Tuple[int, int]:
        return 9, 9,

    def get_initial_agent_position(self) -> Tuple[int, int]:
        gw, gh = self.get_grid_size()
        mid_x = gw // 2
        return mid_x, 7,

    def get_initial_agent_angle(self) -> float:
        return -math.pi / 2

    def get_min_cumulative_reward(self):
        if self.level <= 1:
            return -0.80
        else:
            return -0.85

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
        hint_x = gw - 3 if is_left else 2
        nhs_x = 2 if is_left else gw - 3
        for y in range(3, 5):
            side_elements.append(FlatWorldElement.wall_brick((nhs_x, y)))
            side_elements.append(FlatWorldElement.wall_brick((nhs_x, y)))
        for y in range(3, 5):
            permanent_elements.append(FlatWorldElement.fence_brick((hint_x, y)))
            permanent_elements.append(FlatWorldElement.fence_brick((hint_x, y)))
        # Bottom horizontal
        for x in range(0, 3):
            bottom_elements.append(FlatWorldElement.wall_brick((x, 5)))
            bottom_elements.append(FlatWorldElement.wall_brick((gw - x - 1, 5)))
        # Food
        food_x = 0 if is_left else gw - 1
        food_y = 4
        permanent_elements.append(FlatWorldElement.final_food((food_x, food_y)))
        if self.level == 0:
            se_mask = self.random.uniform(size=len(side_elements)) < 0.20
            s_se = np.array(side_elements)[se_mask]
            selected_elements = bottom_elements + list(s_se)
        elif self.level == 1:
            te_mask = self.random.uniform(size=len(top_elements)) < 0.20
            s_te = np.array(top_elements)[te_mask]
            selected_elements = bottom_elements + side_elements + list(s_te)
        elif self.level == 2:
            ee_mask = self.random.uniform(size=len(entrance_elements)) < 0.5
            s_ee = np.array(entrance_elements)[ee_mask]
            selected_elements = bottom_elements + side_elements + top_elements + list(s_ee)
        else:
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
