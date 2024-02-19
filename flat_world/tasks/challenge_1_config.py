from typing import List, Tuple, Union

import numpy as np

from flat_world.FlatWorldElement import FlatWorldElement
from flat_world.FlatWorldConfig import FlatWorldConfig


class Challenge1Config(FlatWorldConfig):
    def get_min_cumulative_reward(self):
        return -2.0

    def get_initial_agent_position(self) -> Tuple[int, int]:
        return 5, 8,

    def get_initial_agent_angle(self) -> float:
        return 0

    def get_elements(self, agent_x, agent_y) -> List[object]:
        elements = [FlatWorldElement.final_food((9, 8))]
        for b in range(3):
            elements.append(FlatWorldElement.translucent_brick((7, 9 - b)))
            elements.append(FlatWorldElement.fence_brick((7, b)))
        elements.append(FlatWorldElement.fence_brick((7, 3)))
        for b in range(3):
            elements.append(FlatWorldElement.fence_brick((9 - b - 2, 6)))
            elements.append(FlatWorldElement.fence_brick((9 - b - 2, 4)))
        return elements
