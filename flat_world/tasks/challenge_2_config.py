import math
from typing import List, Tuple, Union

import numpy as np

from flat_world.FlatWorldElement import FlatWorldElement
from flat_world.FlatWorldConfig import FlatWorldConfig


class Challenge2Config(FlatWorldConfig):
    def get_min_cumulative_reward(self):
        return -2.0

    def get_initial_agent_position(self) -> Tuple[int, int]:
        return 5, 9,

    def get_initial_agent_angle(self) -> float:
        return -math.pi / 2

    def get_elements(self, agent_x, agent_y) -> List[object]:
        elements = [
            FlatWorldElement.final_food((8, 2)),
            FlatWorldElement.food((2, 2)),
        ]
        for b in range(10):
            if b < 4 or b > 6:
                elements.append(FlatWorldElement.translucent_brick((b, 4)))
        elements.append(FlatWorldElement.box((5, 4)))
        for b in range(2, 5):
            elements.append(FlatWorldElement.fence_brick((4, b)))
            elements.append(FlatWorldElement.fence_brick((6, b)))
        return elements
